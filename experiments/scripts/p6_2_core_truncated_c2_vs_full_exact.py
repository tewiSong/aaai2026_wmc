"""P6.2  Does C2 need the whole ontology exactly, or only each protein's hot core? (plan item (d))

P6.0: at GO bp scale the cold odds mass stays ~1 even with half the vocabulary in the core, so
no truncation certificate is small; but hot cores of hundreds to ~1,000 terms compile in < 1 s.
P6.1: exact core + loopy-BP periphery does not help (errors 0.02-0.15 at the exact theta*,
periphery moment matching does not converge; isotonic ties make the cold couplings near-deterministic).
This experiment measures the accuracy requirement directly on GO cc / GO mf, where the full exact
C2 exists:
  core-truncated C2(tau): per protein, H = {mu*_i >= tau} (ancestor-closed). The model is C2 on H
    only, i.e. the exact posterior conditioned on the cold terms being off: theta*_H by exact moment
    matching of the H-circuit to mu*_H, the same residual features per depth, fitted by L-BFGS on
    the core-restricted joint conditional NLL. Cold terms are predicted by their target mu*_c,
    then the order is enforced (child <= parent, parents first).
  compared with the full exact C2 (P4.5 v2 weights and theta*) and isotonic, on the same
  exploratory split (seed 0), for tau in {0.03, 0.01, 0.003}.
If core-truncated C2 at a tau whose cold mass matches what GO bp cores can reach (P6.0) keeps the
full-C2 gains, GO bp needs only per-protein core circuits.
Output: results/p6_2_core_truncated_c2_<ns>_<input>.json
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
from scipy.special import logit

from consistency import enforce_order_exact, isotonic_dag_hildreth
from learned_constrained import ModelC2, log_partition, moment_match, term_depth
from p4_3_learned_constrained_model_vs_calibration import metrics
from wmc_common import DATA_DIR, RESULTS_DIR, load_theory, write_dimacs
from wmc_inference import Circuit, compile_with_d4

_CORES: dict = {}          # protein row -> (H indices, Circuit or None); filled before the pool forks


def build_core(args):
    k, H, sub, workdir = args
    if len(sub) == 0:
        return k, None
    d = Path(workdir) / f"c{k}"
    d.mkdir(exist_ok=True)
    write_dimacs(d / "c.cnf", len(H), sub)
    compile_with_d4(d / "c.cnf", d / "c.nnf")
    return k, str(d / "c.nnf")


def core_lz(k, th_h):
    circ = _CORES[k][1]
    if circ is None:
        return float(np.sum(np.logaddexp(0.0, th_h))), 1.0 / (1.0 + np.exp(-th_h))
    return log_partition(circ, th_h)


def _mm(args):
    k, mu_h = args
    circ = _CORES[k][1]
    if circ is None:
        return k, logit(np.clip(mu_h, 1e-6, 1 - 1e-6)), 0.0
    th, err = moment_match(circ, mu_h, logit(np.clip(mu_h, 1e-6, 1 - 1e-6)))
    return k, th, err


def _nll_one(args):
    k, th_h, y_h = args
    lz, mu = core_lz(k, th_h)
    return float(th_h @ y_h - lz), y_h - mu


def _pred_one(args):
    k, th_h = args
    return core_lz(k, th_h)[1]


class CoreC2:
    K = 3

    def __init__(self, depth, pool, ridge=1e-3):
        self.depth, self.pool, self.ridge = depth, pool, ridge
        self.D = int(depth.max()) + 1

    def theta(self, w, k, th0, l, m):
        H = _CORES[k][0]
        W = w.reshape(self.D, self.K)[self.depth[H]]
        return th0 + W[:, 0] + W[:, 1] * l[H] + W[:, 2] * m[H]

    def nll_grad(self, w, rows, TH0, L, M, Y):
        tasks = [(k, self.theta(w, k, TH0[k], L[k], M[k]), Y[k][_CORES[k][0]].astype(float)) for k in rows]
        res = self.pool.map(_nll_one, tasks, chunksize=8)
        tot, g = 0.0, np.zeros((self.D, self.K))
        for k, (ll, r) in zip(rows, res):
            H = _CORES[k][0]
            tot -= ll
            dh = self.depth[H]
            for j, f in enumerate((np.ones(len(H)), L[k][H], M[k][H])):
                g[:, j] -= np.bincount(dh, weights=r * f, minlength=self.D)
        n = len(rows)
        return tot / n + self.ridge * np.sum(w ** 2), g.ravel() / n + 2 * self.ridge * w

    def fit(self, rows, TH0, L, M, Y, maxiter=1000):
        res = minimize(self.nll_grad, np.zeros(self.D * self.K), args=(rows, TH0, L, M, Y), jac=True,
                       method="L-BFGS-B", options=dict(maxiter=maxiter))
        self.w = res.x
        return res

    def predict(self, rows, TH0, L, M, MT, edges, w=None):
        w = self.w if w is None else w
        res = self.pool.map(_pred_one, [(k, self.theta(w, k, TH0[k], L[k], M[k])) for k in rows], chunksize=8)
        out = MT[rows].copy()
        for i, (k, mu_h) in enumerate(zip(rows, res)):
            out[i, _CORES[k][0]] = mu_h
        return enforce_order_exact(out, edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--taus", default="0.03,0.01,0.003")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--bootstrap", type=int, default=1000)
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
    if args.limit:
        S, Y, TH_EX = S[: args.limit], Y[: args.limit], TH_EX[: args.limit]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(S))
    fit_idx, ev_idx = perm[: len(S) // 2], perm[len(S) // 2:]
    depth = term_depth(n, edges, root)
    td = Path(tempfile.mkdtemp())
    write_dimacs(td / "t.cnf", n, edges); compile_with_d4(td / "t.cnf", td / "t.nnf")
    circ = Circuit(td / "t.nnf", n)
    mu_unif, _ = circ.marginals(np.full(n, 0.5))
    ISO = isotonic_dag_hildreth(S, edges)[0]
    MT = (1 - args.eps) * ISO + args.eps * mu_unif
    L, M = logit(S), logit(np.clip(MT, 1e-9, 1 - 1e-9))
    Yc = Y[ev_idx][:, cols]

    full = ModelC2(circ, depth)
    full.w = np.array(json.load(open(RESULTS_DIR / f"p4_5_nested_C2_{args.ns}_{args.input}.json"))["C2_fit"]["w"]).ravel()
    outs = {"isotonic": ISO[ev_idx], "C2_full_exact": full.predict(TH_EX[ev_idx], L[ev_idx], M[ev_idx])}
    info = dict(theory=f"dgp2_{args.ns}", input=args.input, stage="exploratory split (seed 0, as P4.5 v2)",
                proteins_fit=int(len(fit_idx)), proteins_eval=int(len(ev_idx)), per_tau={})

    odds = MT / (1 - MT)
    for tau in [float(t) for t in args.taus.split(",")]:
        t0 = time.time()
        _CORES.clear()
        hidx_all, tasks = {}, []
        for k in range(len(S)):
            Hm = MT[k] >= tau
            Hm[root] = True
            H = np.flatnonzero(Hm)
            idx = -np.ones(n, dtype=np.int64); idx[H] = np.arange(len(H))
            inner = Hm[edges[:, 0]] & Hm[edges[:, 1]]
            hidx_all[k] = H
            tasks.append((k, H, idx[edges[inner]], str(td)))
        with Pool(args.workers) as pool:
            built = dict(pool.map(build_core, tasks, chunksize=16))
        for k, H in hidx_all.items():
            _CORES[k] = (H, Circuit(Path(built[k]), len(H)) if built[k] else None)
        t_compile = time.time() - t0
        with Pool(args.workers) as pool:                  # forked after _CORES is filled
            mm = pool.map(_mm, [(k, MT[k][_CORES[k][0]]) for k in range(len(S))], chunksize=8)
            TH0 = {k: th for k, th, _ in mm}
            mm_err = np.array([e for _, _, e in mm])
            model = CoreC2(depth, pool)
            t1 = time.time()
            r = model.fit(list(fit_idx), TH0, L, M, Y)
            t_fit = time.time() - t1
            outs[f"C2_core_tau{tau:g}"] = model.predict(list(ev_idx), TH0, L, M, MT, edges)
        sizes = np.array([len(_CORES[k][0]) for k in range(len(S))])
        lam = np.array([odds[k][np.setdiff1d(np.arange(n), _CORES[k][0])].sum() for k in range(len(S))])
        info["per_tau"][str(tau)] = dict(core_size=np.quantile(sizes, [0.1, 0.5, 0.9, 1.0]).tolist(),
                                         cold_odds_mass=np.quantile(lam, [0.1, 0.5, 0.9]).tolist(),
                                         compile_seconds=round(t_compile, 1), fit_seconds=round(t_fit, 1),
                                         core_mm_err_max=float(mm_err.max()), converged=bool(r.success),
                                         iterations=int(r.nit), nll=float(r.fun), w=model.w.round(4).tolist())
        print(f"tau {tau}: {json.dumps(info['per_tau'][str(tau)])[:400]}", flush=True)

    evm = {k: v[:, cols] for k, v in outs.items()}
    info["metrics"] = {k: metrics(v, Yc) for k, v in evm.items()}
    for k, m in info["metrics"].items():
        print(f"{k:24s} " + "  ".join(f"{a} {b:.5f}" for a, b in m.items()), flush=True)
    names = [k for k in evm if k.startswith("C2_core")]
    pairs = [(a, "C2_full_exact") for a in names] + [(a, "isotonic") for a in names] + [("C2_full_exact", "isotonic")]
    diffs = {f"{a}-{b}": {m: [] for m in ("fmax", "micro_ap", "brier", "logloss")} for a, b in pairs}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(Yc), len(Yc))
        mm_ = {k: metrics(v[idx], Yc[idx]) for k, v in evm.items()}
        for a, b in pairs:
            for m in diffs[f"{a}-{b}"]:
                diffs[f"{a}-{b}"][m].append(mm_[a][m] - mm_[b][m])
    info["diff_95ci"] = {k: {m: np.quantile(v, [0.025, 0.5, 0.975]).tolist() for m, v in d.items()} for k, d in diffs.items()}
    if args.limit:
        return
    json.dump(info, open(RESULTS_DIR / f"p6_2_core_truncated_c2_{args.ns}_{args.input}.json", "w"), indent=1)
    for k, d in info["diff_95ci"].items():  # noqa
        print(f"{k:40s} fmax {np.round(d['fmax'], 4).tolist()}  ap {np.round(d['micro_ap'], 4).tolist()}", flush=True)


if __name__ == "__main__":
    main()
