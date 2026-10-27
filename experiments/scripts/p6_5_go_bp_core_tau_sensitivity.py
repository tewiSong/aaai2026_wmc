"""P6.5  Is GO bp core C2 insensitive to the truncation level? (check on P6.3)

P6.2 showed core-truncated C2 = full exact C2 for tau <= 0.01 on GO cc / GO mf. On GO bp there is
no full exact C2 to compare with, and the cold odds mass at the P6.3 cores is larger (median 3.6
fused, 6.7 CNN). Here, with the P6.3 weights w held fixed, a sample of evaluation proteins is
predicted twice:
  A  the P6.3 adaptive core (same ladder and caps, so the same core as in P6.3)
  B  a strictly larger core: the smallest tau in LOWER below the core-A tau that fits the same caps
     (core <= --max-core terms, d4 <= --d4-timeout s, <= --max-edges circuit edges)
and the two predictions are compared: per-protein max |difference| and paired-bootstrap metric
differences (Fmax, micro AP, Brier, log-loss) on the proteins that got a larger core B.
If the B - A differences are far below the C2 - isotonic gains of P6.3, the GO bp results do not
depend on the truncation level.
Output: results/p6_5_go_bp_core_tau_sensitivity_<input>.json
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

import p6_2_core_truncated_c2_vs_full_exact as p62
import p6_3_go_bp_core_c2_exploratory as p63
from consistency import isotonic_dag_hildreth
from learned_constrained import term_depth
from p4_3_learned_constrained_model_vs_calibration import metrics
from wmc_common import DATA_DIR, RESULTS_DIR, load_theory
from wmc_inference import Circuit

LOWER = [0.001, 0.002, 0.003, 0.005, 0.007]


def build_lower(args):
    k, mt, edges, n, root, tau_a, max_core, max_edges, timeout, workdir = args
    saved = p63.TAUS
    p63.TAUS = [t for t in LOWER if t < tau_a]
    try:
        k, tau, H, nnf, tried, sec = p63.build_adaptive((k, mt, edges, n, root, max_core, max_edges, timeout, workdir))
    finally:
        p63.TAUS = saved
    return k, tau, H, nnf


def run_cores(built, rows, MT, L, M, edges, depth, w, workers):
    p62._CORES.clear()
    for k, H, nnf in built:
        p62._CORES[k] = (H, Circuit(Path(nnf), len(H)) if nnf else None)
    with Pool(workers) as pool:
        t0 = time.time()
        mm = pool.map(p62._mm, [(k, MT[k][p62._CORES[k][0]]) for k in rows], chunksize=1)
        TH0 = {k: th for k, th, _ in mm}
        err = max(e for _, _, e in mm)
        model = p62.CoreC2(depth, pool)
        model.w = w
        P = model.predict(rows, TH0, L, M, MT, edges)
    return P, dict(mm_seconds=round(time.time() - t0, 1), mm_err_max=float(err))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--max-core", type=int, default=1500)
    ap.add_argument("--max-edges", type=int, default=2_000_000)
    ap.add_argument("--d4-timeout", type=float, default=120.0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args = ap.parse_args()

    ns = "go_bp"
    ids, edges = load_theory(f"dgp2_{ns}")
    n = len(ids)
    z = np.load(DATA_DIR / "scores" / f"deepgoplus2_{ns}.npz")
    S, Y, root = z[args.input].astype(float), z["labels"].astype(np.int8), int(z["root_index"])
    cols = np.array([j for j in range(n) if j != root])
    keep = Y[:, cols].sum(1) > 0
    S, Y = np.clip(S[keep], 1e-6, 1 - 1e-6), Y[keep]
    perm = np.random.default_rng(0).permutation(len(S))
    ev_idx = perm[len(S) // 2:]                                   # P6.3 evaluation half
    rows = np.sort(np.random.default_rng(3).choice(ev_idx, min(args.sample, len(ev_idx)), replace=False))
    depth = term_depth(n, edges, root)
    p6_3 = json.load(open(RESULTS_DIR / f"p6_3_go_bp_core_c2_{args.input}.json"))
    w = np.array(p6_3["C2_fit"]["w"]).ravel()

    ISO = np.zeros_like(S)
    ISO[rows] = isotonic_dag_hildreth(S[rows], edges)[0]
    mu_ref = p63.descendant_reference(n, edges)
    MT = (1 - args.eps) * ISO + args.eps * mu_ref
    L, M = logit(S), logit(np.clip(MT, 1e-9, 1 - 1e-9))
    odds = MT / (1 - MT)
    td_a, td_b = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())

    t0 = time.time()
    with Pool(args.workers) as pool:
        A = pool.map(p63.build_adaptive, [(k, MT[k], edges, n, root, args.max_core, args.max_edges, args.d4_timeout,
                                           str(td_a)) for k in rows], chunksize=1)
        tau_a = {k: tau for k, tau, *_ in A}
        B = pool.map(build_lower, [(k, MT[k], edges, n, root, tau_a[k], args.max_core, args.max_edges,
                                    args.d4_timeout, str(td_b)) for k in rows], chunksize=1)
    coreA = {k: (H, nnf) for k, _, H, nnf, _, _ in A}
    coreB = {k: (H, nnf) if tau < 1.0 else coreA[k] for k, tau, H, nnf in B}  # 1.0: nothing lower fits
    tau_b = {k: (tau if tau < 1.0 else tau_a[k]) for k, tau, _, _ in B}
    larger = np.array([tau_b[k] < tau_a[k] for k in rows])
    lam = lambda core, k: float(odds[k].sum() - odds[k][core[k][0]].sum())
    info = dict(theory=f"dgp2_{ns}", input=args.input, proteins=int(len(rows)), lower_taus=LOWER,
                build_seconds=round(time.time() - t0, 1), proteins_with_larger_core=int(larger.sum()),
                tau_a_counts={str(t): int(sum(tau_a[k] == t for k in rows)) for t in sorted(set(tau_a.values()))},
                tau_b_counts={str(t): int(sum(tau_b[k] == t for k in rows[larger])) for t in sorted(set(tau_b[k] for k in rows[larger]))},
                core_size_a=np.quantile([len(coreA[k][0]) for k in rows[larger]], [0.1, 0.5, 0.9, 1.0]).tolist(),
                core_size_b=np.quantile([len(coreB[k][0]) for k in rows[larger]], [0.1, 0.5, 0.9, 1.0]).tolist(),
                cold_odds_mass_a=np.quantile([lam(coreA, k) for k in rows[larger]], [0.1, 0.5, 0.9]).tolist(),
                cold_odds_mass_b=np.quantile([lam(coreB, k) for k in rows[larger]], [0.1, 0.5, 0.9]).tolist())
    print(json.dumps(info), flush=True)

    sub = rows[larger]
    PA, ia = run_cores([(k, *coreA[k]) for k in sub], list(sub), MT, L, M, edges, depth, w, args.workers)
    PB, ib = run_cores([(k, *coreB[k]) for k in sub], list(sub), MT, L, M, edges, depth, w, args.workers)
    info["run_a"], info["run_b"] = ia, ib
    d = np.abs(PB - PA)[:, cols].max(1)
    info["max_abs_diff_per_protein"] = np.quantile(d, [0.5, 0.9, 0.99, 1.0]).tolist()
    Yc = Y[sub][:, cols]
    PAc, PBc, Ic = PA[:, cols], PB[:, cols], ISO[sub][:, cols]
    info["metrics"] = {"core_A_P6.3": metrics(PAc, Yc), "core_B_larger": metrics(PBc, Yc), "isotonic": metrics(Ic, Yc)}
    rng = np.random.default_rng(4)
    diffs = {k: {m: [] for m in ("fmax", "micro_ap", "brier", "logloss")} for k in ("B - A", "A - isotonic")}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(Yc), len(Yc))
        ma, mb, mi = metrics(PAc[idx], Yc[idx]), metrics(PBc[idx], Yc[idx]), metrics(Ic[idx], Yc[idx])
        for m in diffs["B - A"]:
            diffs["B - A"][m].append(mb[m] - ma[m]); diffs["A - isotonic"][m].append(ma[m] - mi[m])
    info["diff_95ci"] = {k: {m: np.quantile(v, [0.025, 0.5, 0.975]).tolist() for m, v in dd.items()} for k, dd in diffs.items()}
    for k, m in info["metrics"].items():
        print(f"{k:16s} " + "  ".join(f"{a} {b:.5f}" for a, b in m.items()), flush=True)
    print("max|B-A| per protein q50/90/99/max", np.round(info["max_abs_diff_per_protein"], 5).tolist(), flush=True)
    for k, dd in info["diff_95ci"].items():
        print(f"{k:14s} " + "  ".join(f"{m} {np.round(v, 5).tolist()}" for m, v in dd.items()), flush=True)
    if args.sample >= 100:
        json.dump(info, open(RESULTS_DIR / f"p6_5_go_bp_core_tau_sensitivity_{args.input}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
