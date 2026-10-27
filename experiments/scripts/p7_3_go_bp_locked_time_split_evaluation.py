"""P7.3  Locked GO bp time-split confirmation (protocol P7_protocol_go_bp_time_split_confirmation.md).

Refuses to run if the protocol file's sha256 differs from results/p7_protocol_lock.txt.
Fits every method on ALL old GO bp proteins (DeepGOPlus 1.0.6 test proteins with a non-root bp
label) and evaluates on the P7 targets (P7.1 labels, P7.2 scores).
Inference for C2 models: per-protein adaptive hot cores exactly as P6.3 (ladder, caps, descendant
reference for the eps shift). Core theta* (old and new proteins) is cached in
data/scores/p7_core_theta_go_bp_<input>.npz and checkpointed, so a stopped job can resume.
Models:
  C2_term_affine (primary)  theta_H = theta*_H + w_{d,0} + w_{d,1} l + w_{d,2} m + b_i + a_i l
                            ridge 1e-3 on w, lambda on (a, b) chosen from LAMBDAS by joint NLL on an
                            inner 80/20 split of the old proteins (seed 1), then refit on all old
  C2 (secondary)            depth-shared residual only (P5 / P6.3 model)
  baselines                 the 9 methods of P5 / P6.3
Output: results/p7_3_eval_go_bp_<input>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

import p6_2_core_truncated_c2_vs_full_exact as p62
import p6_3_go_bp_core_c2_exploratory as p63
from consistency import isotonic_dag_hildreth
from learned_constrained import (apply_depth_platt, apply_neighbor_lr, apply_term_platt, fit_depth_platt,
                                 fit_neighbor_lr, fit_term_platt, neighbor_features, term_depth)
from p4_0_downstream_conditioning_deepgoplus_cc_mf import max_closure
from p4_3_learned_constrained_model_vs_calibration import metrics
from p4_6_ablation_same_features_without_wmc import fit_depth_lr
from wmc_common import DATA_DIR, EXP_DIR, RESULTS_DIR, load_theory
from wmc_inference import Circuit

NS = "go_bp"
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
EPS = 1e-3
FITTED = ["depth_platt_iso", "term_platt_iso", "neighbor_lr_iso", "same_features_lr_iso",
          "theta_star_feature_lr_iso", "whole_vector_pca32_lr_iso"]
NI_MARGIN = 0.005                               # Fmax non-inferiority margin (protocol §7, H3)


def check_lock():
    lock = open(RESULTS_DIR / "p7_protocol_lock.txt").read().split()[2]
    now = hashlib.sha256(open(EXP_DIR / "P7_protocol_go_bp_time_split_confirmation.md", "rb").read()).hexdigest()
    if lock != now:
        raise SystemExit(f"protocol changed after locking ({now} != {lock}); record an amendment first")
    return now


class CoreTermC2(p62.CoreC2):
    """Core C2 with optional per-term affine residual b_i + a_i l_i (term=False: plain core C2)."""

    def __init__(self, depth, n, pool, term, ridge=1e-3, ridge_t=1e-2):
        super().__init__(depth, pool, ridge)
        self.n, self.term, self.ridge_t = n, term, ridge_t
        self.P = self.D * self.K + (2 * n if term else 0)

    def theta(self, x, k, th0, l, m):
        H = p62._CORES[k][0]
        th = super().theta(x[: self.D * self.K], k, th0, l, m)
        if self.term:
            b, a = x[self.D * self.K: self.D * self.K + self.n], x[self.D * self.K + self.n:]
            th = th + b[H] + a[H] * l[H]
        return th

    def nll_grad(self, x, rows, TH0, L, M, Y):
        tasks = [(k, self.theta(x, k, TH0[k], L[k], M[k]), Y[k][p62._CORES[k][0]].astype(float)) for k in rows]
        res = self.pool.map(p62._nll_one, tasks, chunksize=8)
        tot, g = 0.0, np.zeros((self.D, self.K))
        gb, ga = np.zeros(self.n), np.zeros(self.n)
        for k, (ll, r) in zip(rows, res):
            H = p62._CORES[k][0]
            tot -= ll
            dh = self.depth[H]
            for j, f in enumerate((np.ones(len(H)), L[k][H], M[k][H])):
                g[:, j] -= np.bincount(dh, weights=r * f, minlength=self.D)
            if self.term:
                gb[H] -= r; ga[H] -= r * L[k][H]          # H has unique indices
        N = len(rows)
        xw, xt = x[: self.D * self.K], x[self.D * self.K:]
        grad = [g.ravel() / N + 2 * self.ridge * xw]
        loss = tot / N + self.ridge * xw @ xw
        if self.term:
            grad.append(np.r_[gb, ga] / N + 2 * self.ridge_t * xt)
            loss += self.ridge_t * xt @ xt
        return loss, np.concatenate(grad)

    def fit(self, rows, TH0, L, M, Y, x0=None, maxiter=1000):
        x0 = np.zeros(self.P) if x0 is None else x0
        r = minimize(self.nll_grad, x0, args=(rows, TH0, L, M, Y), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=maxiter))
        self.w = r.x
        return r

    def mean_nll(self, rows, TH0, L, M, Y):
        x = self.w
        tasks = [(k, self.theta(x, k, TH0[k], L[k], M[k]), Y[k][p62._CORES[k][0]].astype(float)) for k in rows]
        return -sum(ll for ll, _ in self.pool.map(p62._nll_one, tasks, chunksize=8)) / len(rows)

    def predict(self, rows, TH0, L, M, MT, edges, w=None):
        x = self.w if w is None else w
        res = self.pool.map(p62._pred_one, [(k, self.theta(x, k, TH0[k], L[k], M[k])) for k in rows], chunksize=8)
        out = MT[rows].copy()
        for i, (k, mu_h) in enumerate(zip(rows, res)):
            out[i, p62._CORES[k][0]] = mu_h
        return p62.enforce_order_exact(out, edges)


def core_theta(MT, cache, workers, checkpoint=200):
    """theta*_H for every protein, resuming from the cache where the stored core equals the current one."""
    TH0, done = {}, {}
    if cache.exists():
        c = np.load(cache, allow_pickle=True)
        done = {int(k): (h, t) for k, h, t in zip(c["rows"], c["cores"], c["theta"])}
    todo = []
    for k, (H, _) in p62._CORES.items():
        if k in done and np.array_equal(done[k][0], H):
            TH0[k] = done[k][1]
        else:
            todo.append(k)
    print(f"theta* cache: {len(TH0)} reused, {len(todo)} to compute", flush=True)
    errs, t0 = [], time.time()

    def objs(xs):
        out = np.empty(len(xs), dtype=object)
        for i, x in enumerate(xs):
            out[i] = x
        return out

    def save():
        rows = sorted(TH0)
        np.savez(cache, rows=np.array(rows), cores=objs([p62._CORES[k][0] for k in rows]),
                 theta=objs([TH0[k] for k in rows]))
    with Pool(workers) as pool:
        for i, (k, th, e) in enumerate(pool.imap_unordered(p62._mm, [(k, MT[k][p62._CORES[k][0]]) for k in todo], chunksize=1)):
            TH0[k] = th; errs.append(e)
            if (i + 1) % checkpoint == 0:
                save(); print(f"theta* {i + 1}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    save()
    return TH0, dict(computed=len(todo), seconds=round(time.time() - t0, 1),
                     err_max_computed=float(max(errs)) if errs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["cnn", "fused"], required=True)
    ap.add_argument("--max-core", type=int, default=1500)
    ap.add_argument("--max-edges", type=int, default=2_000_000)
    ap.add_argument("--d4-timeout", type=float, default=120.0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--smoke", type=int, default=0, help="smoke test without lock: this many old and new proteins")
    args = ap.parse_args()
    lock = "smoke-test" if args.smoke else check_lock()

    ids, edges = load_theory(f"dgp2_{NS}")
    n = len(ids)
    old = np.load(DATA_DIR / "scores" / f"deepgoplus2_{NS}.npz")
    S0, Y0, root = old[args.input].astype(float), old["labels"].astype(np.int8), int(old["root_index"])
    cols = np.array([j for j in range(n) if j != root])
    keep = Y0[:, cols].sum(1) > 0
    S0, Y0 = np.clip(S0[keep], 1e-6, 1 - 1e-6), Y0[keep]
    new = np.load(DATA_DIR / "scores" / f"p7_scores_{NS}.npz")
    if args.input == "fused" and not bool(new["fused_valid"]):
        raise SystemExit("gate (b) failed for GO bp: fused input is not part of the confirmation")
    S1, Y1 = np.clip(new[args.input].astype(float), 1e-6, 1 - 1e-6), new["labels"].astype(np.int8)
    assert int(new["root_index"]) == root
    if args.smoke:
        S0, Y0, S1, Y1 = S0[: args.smoke], Y0[: args.smoke], S1[: args.smoke], Y1[: args.smoke]
    N0, N1 = len(S0), len(S1)
    S, Y = np.vstack([S0, S1]), np.vstack([Y0, Y1])
    old_rows, new_rows = list(range(N0)), list(range(N0, N0 + N1))
    depth = term_depth(n, edges, root)
    iso = lambda P: isotonic_dag_hildreth(P, edges)[0]

    ISO = iso(S)
    mu_ref = p63.descendant_reference(n, edges)
    MT = (1 - EPS) * ISO + EPS * mu_ref
    L, M = logit(S), logit(np.clip(MT, 1e-9, 1 - 1e-9))
    info = dict(protocol_sha256=lock, theory=f"dgp2_{NS}", input=args.input, fit_proteins=N0, eval_targets=N1,
                taus=p63.TAUS, max_core=args.max_core, max_edges=args.max_edges, d4_timeout=args.d4_timeout)

    t0 = time.time()
    td = Path(tempfile.mkdtemp())
    with Pool(args.workers) as pool:
        built = pool.map(p63.build_adaptive, [(k, MT[k], edges, n, root, args.max_core, args.max_edges, args.d4_timeout,
                                               str(td)) for k in range(N0 + N1)], chunksize=1)
    p62._CORES.clear()
    tau_used = {}
    for k, tau, H, nnf, tried, sec in built:
        p62._CORES[k] = (H, Circuit(Path(nnf), len(H)) if nnf else None)
        tau_used[k] = tau
    for name, rows in (("old", old_rows), ("new", new_rows)):
        info[f"cores_{name}"] = dict(size=np.quantile([len(p62._CORES[k][0]) for k in rows], [0.1, 0.5, 0.9, 1.0]).tolist(),
                                     tau_counts={str(t): int(sum(tau_used[k] == t for k in rows)) for t in p63.TAUS + [1.0]})
    info["core_build_seconds"] = round(time.time() - t0, 1)
    print("cores", json.dumps({k: info[k] for k in ("cores_old", "cores_new")}), flush=True)

    cache = DATA_DIR / "scores" / f"p7_core_theta_{NS}_{args.input}{'_smoke' if args.smoke else ''}.npz"
    TH0, info["core_theta"] = core_theta(MT, cache, args.workers)
    print("theta*", info["core_theta"], flush=True)

    with Pool(args.workers) as pool:                    # forked after _CORES is filled
        c2 = CoreTermC2(depth, n, pool, term=False)
        t0 = time.time()
        r = c2.fit(old_rows, TH0, L, M, Y)
        info["C2_fit"] = dict(seconds=round(time.time() - t0, 1), converged=bool(r.success), iterations=int(r.nit),
                              nll=float(r.fun), w=c2.w.round(5).tolist())
        print("C2 fit", json.dumps(info["C2_fit"])[:300], flush=True)
        x0 = np.r_[c2.w, np.zeros(2 * n)]

        perm = np.random.default_rng(1).permutation(N0)
        itr, iva = [old_rows[i] for i in perm[: int(0.8 * N0)]], [old_rows[i] for i in perm[int(0.8 * N0):]]
        val = {}
        for lam in LAMBDAS:
            m_ = CoreTermC2(depth, n, pool, term=True, ridge_t=lam)
            t0 = time.time()
            r = m_.fit(itr, TH0, L, M, Y, x0=x0)
            val[str(lam)] = dict(inner_val_nll=m_.mean_nll(iva, TH0, L, M, Y), converged=bool(r.success),
                                 iterations=int(r.nit), seconds=round(time.time() - t0, 1))
            print("lambda", lam, val[str(lam)], flush=True)
        best = min(LAMBDAS, key=lambda l_: val[str(l_)]["inner_val_nll"])
        ta = CoreTermC2(depth, n, pool, term=True, ridge_t=best)
        t0 = time.time()
        r = ta.fit(old_rows, TH0, L, M, Y, x0=x0)
        xt = ta.w[ta.D * ta.K:]
        info["C2_term_affine_fit"] = dict(inner_val=val, lambda_chosen=best, seconds=round(time.time() - t0, 1),
                                          converged=bool(r.success), iterations=int(r.nit), nll=float(r.fun),
                                          term_param_abs_quantiles=np.quantile(np.abs(xt), [0.5, 0.9, 0.99, 1.0]).tolist())
        print("C2_term_affine fit", json.dumps(info["C2_term_affine_fit"])[:400], flush=True)
        c2_new = c2.predict(new_rows, TH0, L, M, MT, edges)
        ta_new = ta.predict(new_rows, TH0, L, M, MT, edges)

    # baselines: fitted on the old proteins, applied to the new targets (P5 / P6.3 definitions)
    o, nw = np.array(old_rows), np.array(new_rows)
    THF = M.copy()
    for k in range(N0 + N1):
        THF[k, p62._CORES[k][0]] = TH0[k]
    Wd, Wt = fit_depth_platt(L[o], Y[o], depth), fit_term_platt(L[o], Y[o])
    Wn = fit_neighbor_lr(neighbor_features(L[o], edges, n), Y[o], depth)
    F = np.stack([np.ones_like(L), L, M], -1)
    W1 = fit_depth_lr(F[o], Y[o], depth)
    W3 = fit_depth_lr(np.concatenate([F[o], THF[o][..., None] / 10.0], -1), Y[o], depth)
    mu_l = L[o].mean(0)
    _, _, Vt = np.linalg.svd(L[o] - mu_l, full_matrices=False)
    Z = (L - mu_l) @ Vt[:32].T / np.sqrt(N0)
    pred = lambda W, X: expit(np.einsum("kni,ni->kn", X, W[depth]))
    W4 = fit_depth_lr(np.concatenate([F[o], np.repeat(Z[o][:, None, :], n, 1)], -1), Y[o], depth)
    outs = {
        "raw": S[nw], "max_closure": max_closure(S[nw], edges, n), "isotonic": ISO[nw],
        "depth_platt_iso": iso(apply_depth_platt(Wd, L[nw], depth)),
        "term_platt_iso": iso(apply_term_platt(Wt, L[nw])),
        "neighbor_lr_iso": iso(apply_neighbor_lr(Wn, neighbor_features(L[nw], edges, n), depth)),
        "same_features_lr_iso": iso(pred(W1, F[nw])),
        "theta_star_feature_lr_iso": iso(pred(W3, np.concatenate([F[nw], THF[nw][..., None] / 10.0], -1))),
        "whole_vector_pca32_lr_iso": iso(pred(W4, np.concatenate([F[nw], np.repeat(Z[nw][:, None, :], n, 1)], -1))),
        "C2": c2_new,
        "C2_term_affine": ta_new,
    }
    evm = {k: v[:, cols] for k, v in outs.items()}
    Yc = Y[nw][:, cols]
    info["metrics"] = {k: metrics(v, Yc) for k, v in evm.items()}
    for k, m in info["metrics"].items():
        print(f"{k:28s} " + "  ".join(f"{a} {b:.5f}" for a, b in m.items()), flush=True)
    rng = np.random.default_rng(0)
    mets = ("fmax", "micro_ap", "brier", "logloss", "ece")
    diffs = {f"{a} - {b}": {m: [] for m in mets} for a in ("C2_term_affine", "C2") for b in evm if b != a}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(Yc), len(Yc))
        mb = {k: metrics(v[idx], Yc[idx]) for k, v in evm.items()}
        for key in diffs:
            a, b = key.split(" - ")
            for m in mets:
                diffs[key][m].append(mb[a][m] - mb[b][m])
    ci = {k: {m: np.quantile(v, [0.025, 0.5, 0.975]).tolist() for m, v in d.items()} for k, d in diffs.items()}
    info["diff_95ci"] = ci
    ap_pos = lambda a, b: ci[f"{a} - {b}"]["micro_ap"][0] > 0
    fmax_ni = lambda a, b: ci[f"{a} - {b}"]["fmax"][0] > -NI_MARGIN
    for a in ("C2_term_affine", "C2"):
        info[f"decision_{a}"] = dict(
            H1_ap_beats_isotonic=bool(ap_pos(a, "isotonic")),
            H2_ap_beats_every_fitted_baseline=bool(all(ap_pos(a, b) for b in FITTED)),
            H2_detail={b: bool(ap_pos(a, b)) for b in FITTED},
            H3_fmax_noninferior_to_isotonic_and_fitted=bool(all(fmax_ni(a, b) for b in ["isotonic"] + FITTED)),
            H3_detail={b: bool(fmax_ni(a, b)) for b in ["isotonic"] + FITTED})
    for k, d in ci.items():
        print(f"{k:44s} fmax {np.round(d['fmax'], 4).tolist()}  ap {np.round(d['micro_ap'], 4).tolist()}", flush=True)
    print(json.dumps({k: v for k, v in info.items() if k.startswith("decision_")}, indent=1), flush=True)
    out = RESULTS_DIR / f"p7_3_eval_{NS}_{args.input}{'_smoke' if args.smoke else ''}.json"
    json.dump(info, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
