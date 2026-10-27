"""P4.5  Corrected learned constrained model C2 (nested with isotonic), stage 1 exploratory split.

Why: P4.4 showed that P4.3's model C (theta = a_d logit s + b_d) could not express the
parameters theta* that reproduce consistent marginals (R^2 = 0.19; theta* - logit s grows by
+3..+8 log-odds with depth), so C was not nested with its baselines and its failure said
nothing about the constrained-model class. C2 starts from theta* for each protein (moment
matching to the interior-shifted isotonic marginals) and learns a depth-shared residual;
at w = 0 it reproduces that isotonic target exactly.

Same split as P4.3 (seed 0, 50/50; exploratory only). Methods on the evaluation half:
  raw, isotonic, isotonic_shifted (= C2 at w=0), C2, term_platt_iso, neighbor_lr_iso,
  depth_platt_iso. Metrics: Fmax, micro AP, Brier, log-loss, ECE; paired bootstrap for C2
  minus each method. theta* for all proteins is cached in data/scores/theta_star_<ns>_<input>.npz.
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
from scipy.special import logit

from consistency import isotonic_dag_hildreth
from learned_constrained import (ModelC2, apply_depth_platt, apply_neighbor_lr, apply_term_platt,
                                 fit_depth_platt, fit_neighbor_lr, fit_term_platt, moment_match,
                                 neighbor_features, term_depth)
from p4_3_learned_constrained_model_vs_calibration import metrics
from wmc_common import DATA_DIR, RESULTS_DIR, load_theory, write_dimacs
from wmc_inference import Circuit, compile_with_d4

_CIRC = None


def _init(nnf, n):
    global _CIRC
    _CIRC = Circuit(Path(nnf), n)


def _mm(args):
    mu_t = args
    th, err = moment_match(_CIRC, mu_t, logit(np.clip(mu_t, 1e-6, 1 - 1e-6)))
    return th, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True)
    ap.add_argument("--input", choices=["cnn", "fused"], required=True)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--limit", type=int, default=0, help="smoke tests: use only this many proteins")
    args = ap.parse_args()

    ids, edges = load_theory(f"dgp2_{args.ns}")
    n = len(ids)
    z = np.load(DATA_DIR / "scores" / f"deepgoplus2_{args.ns}.npz")
    S, Y, root = z[args.input].astype(float), z["labels"].astype(np.int8), int(z["root_index"])
    cols = np.array([j for j in range(n) if j != root])
    keep = Y[:, cols].sum(1) > 0
    S, Y = np.clip(S[keep], 1e-6, 1 - 1e-6), Y[keep]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(S))
    fit_idx, ev_idx = perm[: len(S) // 2], perm[len(S) // 2:]
    if args.limit:
        fit_idx, ev_idx = fit_idx[: args.limit // 2], ev_idx[: args.limit // 2]
    use = np.r_[fit_idx, ev_idx]
    depth = term_depth(n, edges, root)

    td = tempfile.mkdtemp()
    write_dimacs(Path(td) / "t.cnf", n, edges)
    compile_with_d4(Path(td) / "t.cnf", Path(td) / "t.nnf")
    circ = Circuit(Path(td) / "t.nnf", n)
    mu_unif, _ = circ.marginals(np.full(n, 0.5))
    ISO = np.zeros_like(S)
    ISO[use] = isotonic_dag_hildreth(S[use], edges)[0]
    MU_T = (1 - args.eps) * ISO + args.eps * mu_unif

    cache = DATA_DIR / "scores" / f"theta_star_{args.ns}_{args.input}_eps{args.eps:g}_v2.npz"  # v2: stable log Z
    TH0 = np.zeros_like(S); err = np.full(len(S), np.nan)
    if cache.exists() and not args.limit:
        c = np.load(cache); TH0, err = c["theta"], c["err"]
    todo = [i for i in use if np.isnan(err[i])]
    t0 = time.time()
    if todo:
        with Pool(args.workers, initializer=_init, initargs=(str(Path(td) / "t.nnf"), n)) as pool:
            for i, (th, e) in zip(todo, pool.imap(_mm, [MU_T[i] for i in todo], chunksize=4)):
                TH0[i], err[i] = th, e
        if not args.limit:
            np.savez_compressed(cache, theta=TH0, err=err)
    info = dict(theory=f"dgp2_{args.ns}", input=args.input, eps=args.eps, stage="exploratory split (same as P4.3)",
                proteins_fit=int(len(fit_idx)), proteins_eval=int(len(ev_idx)),
                moment_match=dict(seconds=round(time.time() - t0, 1), err_median=float(np.nanmedian(err[use])),
                                  err_p99=float(np.nanquantile(err[use], 0.99)), err_max=float(np.nanmax(err[use]))))
    print(json.dumps(info), flush=True)

    L, M = logit(S), logit(np.clip(MU_T, 1e-9, 1 - 1e-9))
    model = ModelC2(circ, depth, ridge=1e-3)
    t0 = time.time()
    res = model.fit(TH0[fit_idx], L[fit_idx], M[fit_idx], Y[fit_idx], maxiter=args.maxiter)
    nll0 = model.nll_grad(np.zeros_like(model.w), TH0[fit_idx], L[fit_idx], M[fit_idx], Y[fit_idx])[0]
    info["C2_fit"] = dict(seconds=round(time.time() - t0, 1), converged=bool(res.success), iterations=int(res.nit),
                          nll_at_w0=float(nll0), nll_fitted=float(res.fun),
                          w=model.w.reshape(model.D, model.K).round(4).tolist())

    Lf, Yf, Le = L[fit_idx], Y[fit_idx], L[ev_idx]
    Wd, Wt = fit_depth_platt(Lf, Yf, depth), fit_term_platt(Lf, Yf)
    Wn = fit_neighbor_lr(neighbor_features(Lf, edges, n), Yf, depth)
    iso = lambda P: isotonic_dag_hildreth(P, edges)[0]
    outs = {
        "raw": S[ev_idx], "isotonic": ISO[ev_idx],
        "isotonic_shifted_C2_w0": model.predict(TH0[ev_idx], Le, M[ev_idx], w=np.zeros_like(model.w)),
        "C2": model.predict(TH0[ev_idx], Le, M[ev_idx]),
        "depth_platt_iso": iso(apply_depth_platt(Wd, Le, depth)),
        "term_platt_iso": iso(apply_term_platt(Wt, Le)),
        "neighbor_lr_iso": iso(apply_neighbor_lr(Wn, neighbor_features(Le, edges, n), depth)),
    }
    ev = {k: v[:, cols] for k, v in outs.items()}
    Yc = Y[ev_idx][:, cols]
    info["metrics"] = {k: metrics(v, Yc) for k, v in ev.items()}
    for k, m in info["metrics"].items():
        print(f"{k:24s} " + "  ".join(f"{a} {b:.5f}" for a, b in m.items()), flush=True)
    diffs = {k: {m: [] for m in ("fmax", "micro_ap", "brier", "logloss")} for k in ev if k != "C2"}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(Yc), len(Yc))
        mc = metrics(ev["C2"][idx], Yc[idx])
        for k in diffs:
            mk = metrics(ev[k][idx], Yc[idx])
            for m in diffs[k]:
                diffs[k][m].append(mc[m] - mk[m])
    info["C2_minus_other_95ci"] = {k: {m: np.quantile(v, [0.025, 0.5, 0.975]).tolist() for m, v in d.items()}
                                   for k, d in diffs.items()}
    if not args.limit:
        json.dump(info, open(RESULTS_DIR / f"p4_5_nested_C2_{args.ns}_{args.input}.json", "w"), indent=1)
    print(json.dumps(info["C2_fit"]))


if __name__ == "__main__":
    main()
