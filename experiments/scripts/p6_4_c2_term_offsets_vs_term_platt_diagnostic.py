"""P6.4  Why does term Platt match or beat C2 on CNN inputs? (diagnostic, exploratory split)

P5 (GO cc CNN, locked) and P6.3 (GO bp CNN): per-term Platt + isotonic is as good as or better than
C2, while C2 wins on every fused setting and on GO mf. The P4.5 exploratory split reproduces it on
GO cc CNN (C2 - term Platt AP [-0.008, -0.001]). Hypothesis: C2's residual is shared per depth
(D x 3 weights), so it cannot correct systematic per-term miscalibration, which term Platt fixes with
2 weights per term. Variants, all exact (full d4 circuit), same split as P4.5 (seed 0):
  C2                  P4.5 v2 weights (depth-shared residual)
  C2_term_bias        theta = theta* + depth residual + b_i            (ridge lambda on b)
  C2_term_affine      theta = theta* + depth residual + b_i + a_i l_i  (ridge lambda on a, b)
  C2_on_term_platt    C2 (depth-shared) on term-Platt-calibrated scores: theta*, l and m are rebuilt
                      from s' = term Platt(s); fit-half s' are 5-fold out-of-fold, eval-half s' use
                      term Platt fitted on the whole fit half. At w = 0 it is term Platt + isotonic.
lambda for the term parameters is chosen on an inner 80/20 split of the fit half (joint NLL),
then the model is refitted on the whole fit half. The eval half is used only for the final metrics.
If the term-level variants close the gap to term Platt on CNN and keep the gains on fused/mf, the
deficit is the residual's parameter sharing, not the joint (WMC) model.
Also reports per-term bias on the eval half: mean over terms of |mean prediction - label frequency|.
Output: results/p6_4_c2_term_offsets_<ns>_<input>.json
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from consistency import isotonic_dag_hildreth
from learned_constrained import apply_term_platt, fit_term_platt, log_partition, moment_match, term_depth
from p4_3_learned_constrained_model_vs_calibration import metrics
from wmc_common import DATA_DIR, RESULTS_DIR, load_theory, write_dimacs
from wmc_inference import Circuit, compile_with_d4

_CIRC = None
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0]


def _init(nnf, n):
    global _CIRC
    _CIRC = Circuit(Path(nnf), n)


def _nll_chunk(args):
    TH, Y = args
    ll, R = 0.0, np.empty_like(TH)
    for k, (th, y) in enumerate(zip(TH, Y)):
        lz, mu = log_partition(_CIRC, th)
        ll += float(th @ y - lz)
        R[k] = y - mu
    return ll, R


def _pred_chunk(TH):
    return np.stack([log_partition(_CIRC, th)[1] for th in TH])


def _mm(mu_t):
    return moment_match(_CIRC, mu_t, logit(np.clip(mu_t, 1e-6, 1 - 1e-6)))


class TermC2:
    """theta = theta* + w_{d,0} + w_{d,1} l + w_{d,2} m [+ b_i [+ a_i l]]; mode in depth / bias / affine."""

    def __init__(self, depth, n, pool, mode, ridge_w=1e-3, ridge_t=1e-2, chunks=64):
        self.depth, self.n, self.pool, self.mode = depth, n, pool, mode
        self.ridge_w, self.ridge_t, self.chunks = ridge_w, ridge_t, chunks
        self.D = int(depth.max()) + 1
        self.nt = {"depth": 0, "bias": 1, "affine": 2}[mode]
        self.x = np.zeros(self.D * 3 + self.nt * n)

    def theta(self, x, TH0, L, M):
        W = x[: self.D * 3].reshape(self.D, 3)[self.depth]
        TH = TH0 + W[:, 0] + W[:, 1] * L + W[:, 2] * M
        if self.nt >= 1:
            TH = TH + x[self.D * 3: self.D * 3 + self.n]
        if self.nt == 2:
            TH = TH + x[self.D * 3 + self.n:] * L
        return TH

    def _map(self, fn, arrays):
        parts = [np.array_split(a, min(self.chunks, len(a))) for a in arrays]
        return self.pool.map(fn, list(zip(*parts)) if len(arrays) > 1 else parts[0])

    def nll_grad(self, x, TH0, L, M, Y):
        TH = self.theta(x, TH0, L, M)
        res = self._map(_nll_chunk, [TH, Y])
        ll = sum(r[0] for r in res)
        R = np.concatenate([r[1] for r in res])
        N, D = len(TH), self.D
        g = [-np.bincount(self.depth, weights=(R * f).sum(0), minlength=D) for f in (1.0, L, M)]
        g = [np.stack(g, axis=1).ravel()]
        if self.nt >= 1:
            g.append(-R.sum(0))
        if self.nt == 2:
            g.append(-(R * L).sum(0))
        g = np.concatenate(g) / N
        xw, xt = x[: D * 3], x[D * 3:]
        loss = -ll / N + self.ridge_w * xw @ xw + self.ridge_t * xt @ xt
        g[: D * 3] += 2 * self.ridge_w * xw
        g[D * 3:] += 2 * self.ridge_t * xt
        return loss, g

    def fit(self, TH0, L, M, Y, x0=None, maxiter=1000):
        x0 = np.zeros_like(self.x) if x0 is None else x0
        r = minimize(self.nll_grad, x0, args=(TH0, L, M, Y), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=maxiter))
        self.x = r.x
        return r

    def mean_nll(self, TH0, L, M, Y, x=None):
        TH = self.theta(self.x if x is None else x, TH0, L, M)
        return -sum(r[0] for r in self._map(_nll_chunk, [TH, Y])) / len(TH)

    def predict(self, TH0, L, M):
        return np.concatenate(self._map(_pred_chunk, [self.theta(self.x, TH0, L, M)]))


def term_bias(P, Y):
    has = Y.sum(0) > 0
    d = P.mean(0) - Y.mean(0)
    return dict(abs_bias_all=float(np.abs(d).mean()), abs_bias_pos_terms=float(np.abs(d[has]).mean()),
                abs_log_ratio_pos_terms=float(np.abs(np.log(P.mean(0)[has] / Y.mean(0)[has])).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True)
    ap.add_argument("--input", choices=["cnn", "fused"], required=True)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only this many proteins")
    args = ap.parse_args()

    ids, edges = load_theory(f"dgp2_{args.ns}")
    n = len(ids)
    z = np.load(DATA_DIR / "scores" / f"deepgoplus2_{args.ns}.npz")
    S, Y, root = z[args.input].astype(float), z["labels"].astype(np.int8), int(z["root_index"])
    cols = np.array([j for j in range(n) if j != root])
    keep = Y[:, cols].sum(1) > 0
    S, Y = np.clip(S[keep], 1e-6, 1 - 1e-6), Y[keep]
    TH_EX = np.load(DATA_DIR / "scores" / f"theta_star_{args.ns}_{args.input}_eps{args.eps:g}_v2.npz")["theta"]
    assert len(TH_EX) == len(S)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(S))
    fit_idx, ev_idx = perm[: len(S) // 2], perm[len(S) // 2:]
    if args.limit:
        fit_idx, ev_idx = fit_idx[: args.limit // 2], ev_idx[: args.limit // 2]
    depth = term_depth(n, edges, root)
    Yf = Y[fit_idx].astype(float)
    Yc = Y[ev_idx][:, cols]

    td = Path(tempfile.mkdtemp())
    write_dimacs(td / "t.cnf", n, edges); compile_with_d4(td / "t.cnf", td / "t.nnf")
    circ = Circuit(td / "t.nnf", n)
    mu_unif, _ = circ.marginals(np.full(n, 0.5))
    iso = lambda P: isotonic_dag_hildreth(P, edges)[0]
    shift = lambda P: (1 - args.eps) * P + args.eps * mu_unif
    lgt = lambda P: logit(np.clip(P, 1e-9, 1 - 1e-9))

    L = logit(S)
    ISO = np.zeros_like(S)
    use = np.r_[fit_idx, ev_idx]
    ISO[use] = iso(S[use])
    M = lgt(shift(ISO))
    info = dict(theory=f"dgp2_{args.ns}", input=args.input, terms=n, stage="exploratory split (seed 0, as P4.5 v2)",
                proteins_fit=int(len(fit_idx)), proteins_eval=int(len(ev_idx)), lambdas=LAMBDAS, variants={})

    # term Platt on the whole fit half (eval-half baseline and eval-half inputs of C2_on_term_platt)
    Wt = fit_term_platt(L[fit_idx], Y[fit_idx])
    outs = {"isotonic": ISO[ev_idx], "term_platt_iso": iso(apply_term_platt(Wt, L[ev_idx]))}
    # 5-fold out-of-fold term Platt scores for the fit half
    S_tp = np.zeros_like(S)
    folds = np.array_split(np.random.default_rng(2).permutation(len(fit_idx)), 5)
    for f in folds:
        tr = np.setdiff1d(np.arange(len(fit_idx)), f)
        S_tp[fit_idx[f]] = apply_term_platt(fit_term_platt(L[fit_idx[tr]], Y[fit_idx[tr]]), L[fit_idx[f]])
    S_tp[ev_idx] = apply_term_platt(Wt, L[ev_idx])
    S_tp = np.clip(S_tp, 1e-6, 1 - 1e-6)
    ISO_tp = np.zeros_like(S)
    ISO_tp[use] = iso(S_tp[use])

    inner = np.random.default_rng(1).permutation(len(fit_idx))
    itr, iva = fit_idx[inner[: int(0.8 * len(inner))]], fit_idx[inner[int(0.8 * len(inner)):]]

    with Pool(args.workers, initializer=_init, initargs=(str(td / "t.nnf"), n)) as pool:
        t0 = time.time()
        TH_TP = np.zeros_like(S)
        mm = pool.map(_mm, list(shift(ISO_tp[use])), chunksize=4)
        TH_TP[use] = np.stack([m[0] for m in mm])
        info["term_platt_theta_star"] = dict(seconds=round(time.time() - t0, 1),
                                             err_max=float(max(m[1] for m in mm)))
        print("theta* (term Platt inputs)", info["term_platt_theta_star"], flush=True)

        c2 = TermC2(depth, n, pool, "depth")
        if args.limit:
            c2.fit(TH_EX[fit_idx], L[fit_idx], M[fit_idx], Yf, maxiter=args.maxiter)
        else:
            c2.x = np.array(json.load(open(RESULTS_DIR / f"p4_5_nested_C2_{args.ns}_{args.input}.json"))["C2_fit"]["w"]).ravel()
        outs["C2"] = c2.predict(TH_EX[ev_idx], L[ev_idx], M[ev_idx])

        for mode in ("bias", "affine"):
            t0 = time.time()
            val = {}
            for lam in LAMBDAS:
                m_ = TermC2(depth, n, pool, mode, ridge_t=lam)
                r = m_.fit(TH_EX[itr], L[itr], M[itr], Y[itr].astype(float), x0=np.r_[c2.x, np.zeros(m_.nt * n)],
                           maxiter=args.maxiter)
                val[lam] = dict(inner_val_nll=m_.mean_nll(TH_EX[iva], L[iva], M[iva], Y[iva].astype(float)),
                                converged=bool(r.success), iterations=int(r.nit))
                print(mode, lam, val[lam], flush=True)
            best = min(LAMBDAS, key=lambda l_: val[l_]["inner_val_nll"])
            m_ = TermC2(depth, n, pool, mode, ridge_t=best)
            r = m_.fit(TH_EX[fit_idx], L[fit_idx], M[fit_idx], Yf, x0=np.r_[c2.x, np.zeros(m_.nt * n)],
                       maxiter=args.maxiter)
            xt = m_.x[m_.D * 3:]
            info["variants"][f"C2_term_{mode}"] = dict(
                inner_val=val, lambda_chosen=best, converged=bool(r.success), iterations=int(r.nit),
                seconds=round(time.time() - t0, 1), nll=float(r.fun),
                term_param_abs_quantiles=np.quantile(np.abs(xt), [0.5, 0.9, 0.99, 1.0]).tolist())
            print(mode, json.dumps(info["variants"][f"C2_term_{mode}"])[:400], flush=True)
            outs[f"C2_term_{mode}"] = m_.predict(TH_EX[ev_idx], L[ev_idx], M[ev_idx])

        t0 = time.time()
        L_tp, M_tp = logit(S_tp), lgt(shift(ISO_tp))
        m_ = TermC2(depth, n, pool, "depth")
        r = m_.fit(TH_TP[fit_idx], L_tp[fit_idx], M_tp[fit_idx], Yf, maxiter=args.maxiter)
        info["variants"]["C2_on_term_platt"] = dict(converged=bool(r.success), iterations=int(r.nit),
                                                     seconds=round(time.time() - t0, 1), nll=float(r.fun))
        print("on_term_platt", info["variants"]["C2_on_term_platt"], flush=True)
        outs["C2_on_term_platt"] = m_.predict(TH_TP[ev_idx], L_tp[ev_idx], M_tp[ev_idx])
        m_.x = np.zeros_like(m_.x)
        outs["term_platt_iso_shifted_w0"] = m_.predict(TH_TP[ev_idx], L_tp[ev_idx], M_tp[ev_idx])

    evm = {k: v[:, cols] for k, v in outs.items()}
    info["metrics"] = {k: dict(**metrics(v, Yc), **term_bias(v, Yc)) for k, v in evm.items()}
    for k, m in info["metrics"].items():
        print(f"{k:26s} " + "  ".join(f"{a} {b:.5f}" for a, b in m.items()), flush=True)
    news = ["C2_term_bias", "C2_term_affine", "C2_on_term_platt"]
    pairs = [(a, b) for a in news for b in ("term_platt_iso", "C2", "isotonic")] + [("C2", "term_platt_iso")]
    diffs = {f"{a} - {b}": {m: [] for m in ("fmax", "micro_ap", "brier", "logloss")} for a, b in pairs}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(Yc), len(Yc))
        mb = {k: metrics(v[idx], Yc[idx]) for k, v in evm.items() if k != "term_platt_iso_shifted_w0"}
        for a, b in pairs:
            for m in diffs[f"{a} - {b}"]:
                diffs[f"{a} - {b}"][m].append(mb[a][m] - mb[b][m])
    info["diff_95ci"] = {k: {m: np.quantile(v, [0.025, 0.5, 0.975]).tolist() for m, v in d.items()} for k, d in diffs.items()}
    for k, d in info["diff_95ci"].items():
        print(f"{k:36s} fmax {np.round(d['fmax'], 4).tolist()}  ap {np.round(d['micro_ap'], 4).tolist()}  "
              f"logloss {np.round(d['logloss'], 5).tolist()}", flush=True)
    if not args.limit:
        json.dump(info, open(RESULTS_DIR / f"p6_4_c2_term_offsets_{args.ns}_{args.input}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
