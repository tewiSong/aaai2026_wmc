"""P6.3  C2 at GO bp scale with per-protein hot-core circuits (plan item (d)), exploratory split.

P6.2: on GO cc / GO mf, C2 restricted to H = {mu*_i >= tau} (exact posterior given the cold terms
off) is indistinguishable from the full exact C2 for tau <= 0.01, although the cold odds mass is ~1.
P6.0: the full GO bp vocabulary theory does not compile, but its hot cores at tau = 0.01 compile in
seconds (19/20; median 2.3 s CNN). Here every protein gets an adaptive core: the smallest tau on
the ladder TAUS whose core has at most --max-core terms, compiles within --d4-timeout and gives a
circuit of at most --max-edges edges (a pass costs ~0.6 us per edge).
Interior reference for the eps shift: exact p = 0.5 marginals are not computable on GO bp, so
mu_ref_i = (|descendants(i)| + 1) / (n + 1), strictly ordered along every edge (cores only see
values >= tau >> eps, so this choice is immaterial there).
Methods (same split, features and hyperparameters as P4.5 v2 / P4.6 v2 / P5): raw, isotonic,
depth/term Platt + iso, neighbour LR + iso, same-features LR + iso, theta*-feature LR + iso
(theta* = core theta*, logit(mu*) outside), whole-vector PCA32 LR + iso, and core C2.
Output: results/p6_3_go_bp_core_c2_<input>.json
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
from scipy.special import expit, logit

import p6_2_core_truncated_c2_vs_full_exact as p62
from consistency import isotonic_dag_hildreth
from learned_constrained import (apply_depth_platt, apply_neighbor_lr, apply_term_platt, fit_depth_platt,
                                 fit_neighbor_lr, fit_term_platt, neighbor_features, term_depth)
from p4_0_downstream_conditioning_deepgoplus_cc_mf import max_closure
from p4_3_learned_constrained_model_vs_calibration import metrics
from p4_6_ablation_same_features_without_wmc import fit_depth_lr
from wmc_common import DATA_DIR, RESULTS_DIR, load_theory, write_dimacs
from wmc_inference import Circuit, compile_with_d4

TAUS = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]
# 2026-10-24: ladder extended beyond 0.3 after the CNN run stopped on a protein whose CNN scores are
# >= 0.3 on 1,593 terms; levels <= 0.3 are unchanged, so every protein of the fused run keeps its core.
# If no level fits, the core is the root alone (C2 then predicts the target mu* for that protein).


def descendant_reference(n, edges):
    children = [[] for _ in range(n)]
    parents = [[] for _ in range(n)]
    for c, q in edges:
        children[q].append(int(c)); parents[c].append(int(q))
    desc = [None] * n
    pending = np.array([len(children[v]) for v in range(n)])
    stack = [v for v in range(n) if pending[v] == 0]
    while stack:
        v = stack.pop()
        s = set()
        for c in children[v]:
            s |= desc[c]; s.add(c)
        desc[v] = s
        for q in parents[v]:
            pending[q] -= 1
            if pending[q] == 0:
                stack.append(q)
    return np.array([(len(d) + 1) / (n + 1) for d in desc])


def build_adaptive(args):
    k, mt, edges, n, root, max_core, max_edges, timeout, workdir = args
    d = Path(workdir) / f"c{k}"
    d.mkdir(exist_ok=True)
    tried = []
    for tau in TAUS:
        Hm = mt >= tau
        Hm[root] = True
        H = np.flatnonzero(Hm)
        if len(H) > max_core:
            tried.append((tau, int(len(H)), "too_large")); continue
        idx = -np.ones(n, dtype=np.int64); idx[H] = np.arange(len(H))
        inner = Hm[edges[:, 0]] & Hm[edges[:, 1]]
        if not inner.any():
            return k, tau, H, None, tried, 0.0
        write_dimacs(d / "c.cnf", len(H), idx[edges[inner]])
        try:
            r = compile_with_d4(d / "c.cnf", d / "c.nnf", timeout_s=timeout)
            with open(d / "c.nnf") as fh:
                n_edges = sum(1 for line in fh if line[:1].isdigit())
            if n_edges > max_edges:                    # one circuit pass costs ~0.6 us per edge (P6.3 smoke test)
                tried.append((tau, int(len(H)), f"circuit_edges_{n_edges}")); continue
            return k, tau, H, str(d / "c.nnf"), tried, r["seconds"]
        except Exception as e:
            tried.append((tau, int(len(H)), type(e).__name__))
    H = np.array([root])
    return k, 1.0, H, None, tried + [("root_only", 1, "fallback")], 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--max-core", type=int, default=1500)
    ap.add_argument("--max-edges", type=int, default=2_000_000)
    ap.add_argument("--d4-timeout", type=float, default=120.0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ns = "go_bp"
    ids, edges = load_theory(f"dgp2_{ns}")
    n = len(ids)
    z = np.load(DATA_DIR / "scores" / f"deepgoplus2_{ns}.npz")
    S, Y, root = z[args.input].astype(float), z["labels"].astype(np.int8), int(z["root_index"])
    cols = np.array([j for j in range(n) if j != root])
    keep = Y[:, cols].sum(1) > 0
    S, Y = np.clip(S[keep], 1e-6, 1 - 1e-6), Y[keep]
    if args.limit:
        S, Y = S[: args.limit], Y[: args.limit]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(S))
    fit_idx, ev_idx = perm[: len(S) // 2], perm[len(S) // 2:]
    depth = term_depth(n, edges, root)
    td = Path(tempfile.mkdtemp())

    t0 = time.time()
    ISO = isotonic_dag_hildreth(S, edges)[0]
    mu_ref = descendant_reference(n, edges)
    assert np.all(mu_ref[edges[:, 0]] < mu_ref[edges[:, 1]])
    MT = (1 - args.eps) * ISO + args.eps * mu_ref
    L, M = logit(S), logit(np.clip(MT, 1e-9, 1 - 1e-9))
    info = dict(theory=f"dgp2_{ns}", terms=n, input=args.input, stage="exploratory split (seed 0)",
                proteins_fit=int(len(fit_idx)), proteins_eval=int(len(ev_idx)), taus=TAUS, max_core=args.max_core, max_edges=args.max_edges,
                d4_timeout=args.d4_timeout, isotonic_seconds=round(time.time() - t0, 1))

    t0 = time.time()
    tasks = [(k, MT[k], edges, n, root, args.max_core, args.max_edges, args.d4_timeout, str(td)) for k in range(len(S))]
    with Pool(args.workers) as pool:
        built = pool.map(build_adaptive, tasks, chunksize=4)
    p62._CORES.clear()
    tau_used, comp_s, fallbacks = [], [], 0
    for k, tau, H, nnf, tried, sec in built:
        p62._CORES[k] = (H, Circuit(Path(nnf), len(H)) if nnf else None)
        tau_used.append(tau); comp_s.append(sec); fallbacks += int(bool(tried))
    sizes = np.array([len(p62._CORES[k][0]) for k in range(len(S))])
    odds = MT / (1 - MT)
    lam = np.array([odds[k].sum() - odds[k][p62._CORES[k][0]].sum() for k in range(len(S))])
    info["cores"] = dict(build_seconds=round(time.time() - t0, 1), size=np.quantile(sizes, [0.1, 0.5, 0.9, 1.0]).tolist(),
                         tau_counts={str(t): int(np.sum(np.array(tau_used) == t)) for t in TAUS + [1.0]},
                         proteins_with_fallback=fallbacks, compile_seconds=np.quantile(comp_s, [0.5, 0.9, 1.0]).tolist(),
                         cold_odds_mass=np.quantile(lam, [0.1, 0.5, 0.9]).tolist(),
                         circuit_edges=np.quantile([p62._CORES[k][1].n_edges if p62._CORES[k][1] else 0
                                                    for k in range(len(S))], [0.5, 0.9, 1.0]).tolist())
    print("cores", json.dumps(info["cores"]), flush=True)

    with Pool(args.workers) as pool:
        t0 = time.time()
        mm = pool.map(p62._mm, [(k, MT[k][p62._CORES[k][0]]) for k in range(len(S))], chunksize=4)
        TH0 = {k: th for k, th, _ in mm}
        mm_err = np.array([e for _, _, e in mm])
        info["core_moment_matching"] = dict(seconds=round(time.time() - t0, 1), err_median=float(np.median(mm_err)),
                                            err_max=float(mm_err.max()))
        print("mm", json.dumps(info["core_moment_matching"]), flush=True)
        model = p62.CoreC2(depth, pool)
        t0 = time.time()
        r = model.fit(list(fit_idx), TH0, L, M, Y)
        info["C2_fit"] = dict(seconds=round(time.time() - t0, 1), converged=bool(r.success), iterations=int(r.nit),
                              nll=float(r.fun), w=model.w.round(4).tolist())
        print("fit", json.dumps(info["C2_fit"])[:300], flush=True)
        c2_ev = model.predict(list(ev_idx), TH0, L, M, MT, edges)

    THF = M.copy()                                   # theta*-feature: core theta*, logit(mu*) on cold terms
    for k in range(len(S)):
        THF[k, p62._CORES[k][0]] = TH0[k]
    iso = lambda P: isotonic_dag_hildreth(P, edges)[0]
    fi, ev = fit_idx, ev_idx
    Wd, Wt = fit_depth_platt(L[fi], Y[fi], depth), fit_term_platt(L[fi], Y[fi])
    Wn = fit_neighbor_lr(neighbor_features(L[fi], edges, n), Y[fi], depth)
    F = np.stack([np.ones_like(L), L, M], -1)
    W1 = fit_depth_lr(F[fi], Y[fi], depth)
    G = np.concatenate([F, THF[..., None] / 10.0], -1)
    W3 = fit_depth_lr(G[fi], Y[fi], depth)
    del G
    mu_l = L[fi].mean(0)
    _, _, Vt = np.linalg.svd(L[fi] - mu_l, full_matrices=False)
    Z = (L - mu_l) @ Vt[:32].T / np.sqrt(len(fi))
    pred = lambda W, X: expit(np.einsum("kni,ni->kn", X, W[depth]))
    W4 = fit_depth_lr(np.concatenate([F[fi], np.repeat(Z[fi][:, None, :], n, 1)], -1), Y[fi], depth)
    H_ev = np.concatenate([F[ev], np.repeat(Z[ev][:, None, :], n, 1)], -1)
    outs = {
        "raw": S[ev], "max_closure": max_closure(S[ev], edges, n), "isotonic": ISO[ev],
        "depth_platt_iso": iso(apply_depth_platt(Wd, L[ev], depth)),
        "term_platt_iso": iso(apply_term_platt(Wt, L[ev])),
        "neighbor_lr_iso": iso(apply_neighbor_lr(Wn, neighbor_features(L[ev], edges, n), depth)),
        "same_features_lr_iso": iso(pred(W1, F[ev])),
        "theta_star_feature_lr_iso": iso(pred(W3, np.concatenate([F[ev], THF[ev][..., None] / 10.0], -1))),
        "whole_vector_pca32_lr_iso": iso(pred(W4, H_ev)),
        "C2_core": c2_ev,
    }
    del H_ev
    evm = {k: v[:, cols] for k, v in outs.items()}
    Yc = Y[ev][:, cols]
    info["metrics"] = {k: metrics(v, Yc) for k, v in evm.items()}
    for k, m in info["metrics"].items():
        print(f"{k:28s} " + "  ".join(f"{a} {b:.5f}" for a, b in m.items()), flush=True)
    diffs = {k: {m: [] for m in ("fmax", "micro_ap", "brier", "logloss", "ece")} for k in evm if k != "C2_core"}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(Yc), len(Yc))
        mc = metrics(evm["C2_core"][idx], Yc[idx])
        for k in diffs:
            mk = metrics(evm[k][idx], Yc[idx])
            for m in diffs[k]:
                diffs[k][m].append(mc[m] - mk[m])
    info["C2_minus_other_95ci"] = {k: {m: np.quantile(v, [0.025, 0.5, 0.975]).tolist() for m, v in d.items()}
                                   for k, d in diffs.items()}
    for k, d in info["C2_minus_other_95ci"].items():
        print(f"C2_core - {k:26s} fmax {np.round(d['fmax'], 4).tolist()}  ap {np.round(d['micro_ap'], 4).tolist()}", flush=True)
    if not args.limit:
        json.dump(info, open(RESULTS_DIR / f"p6_3_go_bp_core_c2_{args.input}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
