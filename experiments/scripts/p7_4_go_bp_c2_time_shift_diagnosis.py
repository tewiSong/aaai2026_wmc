"""P7.4  Why did core C2 fail the locked GO bp confirmation (P7.3)? Post-hoc diagnosis, exploratory.

Not part of the locked P7 protocol; nothing here changes the P7.3 decision. Uses the stored C2
weights (P6.3: fitted on old half A; P7.3: fitted on all old proteins) and the cached core theta*.
Proteins: the P6.3 evaluation half B of the old GO bp proteins (held out for the P6.3 weights) and
the P7 new targets. Predictions are made with the P6.3 adaptive-core rule.
  Q1  time shift or overfit: the P6.3 weights on old half B (held out) vs on the new targets
  Q2  where AP is lost: core vs cold entries; mean within-protein AP vs pooled micro AP; by depth
  Q3  covariate shift: score / core statistics, old vs new; per-protein C2 - isotonic AP against
      cold odds mass, core size, max score, label count and DIAMOND hit
  Q4  scale: predicted mass / label count on core entries
Outputs: results/p7_4_go_bp_c2_time_shift_diagnosis_<input>.json,
         data/scores/p7_4_predictions_go_bp_<input>.npz
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
from scipy.stats import spearmanr

import p6_2_core_truncated_c2_vs_full_exact as p62
import p6_3_go_bp_core_c2_exploratory as p63
from consistency import isotonic_dag_hildreth
from learned_constrained import term_depth
from p4_0_downstream_conditioning_deepgoplus_cc_mf import fmax
from p4_1b_downstream_consistency_methods_deepgoplus import micro_ap
from p5_2b_fused_scores_with_diamond_gate import DGP
from wmc_common import DATA_DIR, RESULTS_DIR, load_theory
from wmc_inference import Circuit

NS = "go_bp"
EPS = 1e-3
DEPTH_BINS = [(1, 2), (3, 4), (5, 6), (7, 99)]


def diamond_hit(fused, cnn, alpha):
    return np.abs(fused - np.clip((1 - alpha) * cnn, 1e-6, 1 - 1e-6)).max(1) > 1e-5


def per_protein_ap(P, Y):
    return np.array([micro_ap(p[None], y[None]) if y.any() else np.nan for p, y in zip(P, Y)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["cnn", "fused"], required=True)
    ap.add_argument("--max-core", type=int, default=1500)
    ap.add_argument("--max-edges", type=int, default=2_000_000)
    ap.add_argument("--d4-timeout", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args = ap.parse_args()

    ids, edges = load_theory(f"dgp2_{NS}")
    n = len(ids)
    old = np.load(DATA_DIR / "scores" / f"deepgoplus2_{NS}.npz")
    new = np.load(DATA_DIR / "scores" / f"p7_scores_{NS}.npz")
    root = int(old["root_index"])
    cols = np.array([j for j in range(n) if j != root])
    Y0 = old["labels"].astype(np.int8)
    keep = Y0[:, cols].sum(1) > 0
    clip = lambda a: np.clip(a.astype(float), 1e-6, 1 - 1e-6)
    S0, Y0 = clip(old[args.input])[keep], Y0[keep]
    S1, Y1 = clip(new[args.input]), new["labels"].astype(np.int8)
    alpha = json.load(open(DGP / "metadata" / "last_release.json"))["alphas"]["bp"]
    hit = np.r_[diamond_hit(clip(old["fused"])[keep], clip(old["cnn"])[keep], alpha),
                diamond_hit(clip(new["fused"]), clip(new["cnn"]), alpha)]
    N0, N1 = len(S0), len(S1)
    S, Y = np.vstack([S0, S1]), np.vstack([Y0, Y1])
    ev = np.sort(np.random.default_rng(0).permutation(N0)[N0 // 2:])         # P6.3 evaluation half
    groups = {"old_half_B": ev, "new_targets": np.arange(N0, N0 + N1)}
    rows = np.r_[ev, groups["new_targets"]]
    depth = term_depth(n, edges, root)

    ISO = isotonic_dag_hildreth(S, edges)[0]
    MT = (1 - EPS) * ISO + EPS * p63.descendant_reference(n, edges)
    L, M = logit(S), logit(np.clip(MT, 1e-9, 1 - 1e-9))
    odds = MT / (1 - MT)

    t0 = time.time()
    td = Path(tempfile.mkdtemp())
    with Pool(args.workers) as pool:
        built = pool.map(p63.build_adaptive, [(k, MT[k], edges, n, root, args.max_core, args.max_edges, args.d4_timeout,
                                               str(td)) for k in rows], chunksize=1)
    p62._CORES.clear()
    tau = {}
    for k, t, H, nnf, _, _ in built:
        p62._CORES[k] = (H, Circuit(Path(nnf), len(H)) if nnf else None)
        tau[k] = t
    info = dict(theory=f"dgp2_{NS}", input=args.input, stage="post-hoc diagnosis of P7.3 (exploratory)",
                proteins={g: int(len(r)) for g, r in groups.items()}, core_build_seconds=round(time.time() - t0, 1))

    c = np.load(DATA_DIR / "scores" / f"p7_core_theta_{NS}_{args.input}.npz", allow_pickle=True)
    cached = {int(k): (h, t) for k, h, t in zip(c["rows"], c["cores"], c["theta"])}
    TH0 = {k: cached[k][1] for k in rows if k in cached and np.array_equal(cached[k][0], p62._CORES[k][0])}
    todo = [k for k in rows if k not in TH0]
    with Pool(args.workers) as pool:
        for k, th, _ in pool.map(p62._mm, [(k, MT[k][p62._CORES[k][0]]) for k in todo], chunksize=1):
            TH0[k] = th
        W = {"C2_P6.3_w": np.array(json.load(open(RESULTS_DIR / f"p6_3_go_bp_core_c2_{args.input}.json"))["C2_fit"]["w"]).ravel(),
             "C2_P7.3_w": np.array(json.load(open(RESULTS_DIR / f"p7_3_eval_{NS}_{args.input}.json"))["C2_fit"]["w"]).ravel()}
        model = p62.CoreC2(depth, pool)
        P = {"isotonic": ISO[rows]}
        for name, w in W.items():
            P[name] = model.predict(list(rows), TH0, L, M, MT, edges, w=w)
    info["theta_star_recomputed"] = len(todo)
    pos = {k: i for i, k in enumerate(rows)}

    core = np.zeros((len(rows), n), bool)
    for i, k in enumerate(rows):
        core[i, p62._CORES[k][0]] = True
    lam = np.array([odds[k].sum() - odds[k][p62._CORES[k][0]].sum() for k in rows])

    out = {}
    for g, gr in groups.items():
        idx = np.array([pos[k] for k in gr])
        Yg, Cg = Y[gr][:, cols], core[idx][:, cols]
        dg = depth[cols]
        res = dict(covariates=dict(
            core_size=np.quantile(core[idx].sum(1), [0.1, 0.5, 0.9]).tolist(),
            tau_0p01_fraction=float(np.mean([tau[k] == 0.01 for k in gr])),
            cold_odds_mass=np.quantile(lam[idx], [0.1, 0.5, 0.9]).tolist(),
            max_score=np.quantile(S[gr][:, cols].max(1), [0.1, 0.5, 0.9]).tolist(),
            terms_score_gt_0p1=np.quantile((S[gr][:, cols] > 0.1).sum(1), [0.1, 0.5, 0.9]).tolist(),
            labels_per_protein=np.quantile(Yg.sum(1), [0.1, 0.5, 0.9]).tolist(),
            positives_in_core_fraction=float(Yg[Cg].sum() / Yg.sum()),
            diamond_hit_fraction=float(hit[gr].mean())))
        m = {}
        ppa = {}
        for name, Pm in P.items():
            Pg = Pm[idx][:, cols]
            ppa[name] = per_protein_ap(Pg, Yg)
            m[name] = dict(fmax=fmax(Pg, Yg), micro_ap=micro_ap(Pg, Yg),
                           micro_ap_core_entries=micro_ap(Pg[Cg][None], Yg[Cg][None]),
                           micro_ap_cold_entries=micro_ap(Pg[~Cg][None], Yg[~Cg][None]),
                           mean_per_protein_ap=float(np.nanmean(ppa[name])),
                           pred_over_label_mass_core=float(Pg[Cg].sum() / Yg[Cg].sum()),
                           pred_over_label_mass_cold=float(Pg[~Cg].sum() / max(Yg[~Cg].sum(), 1)),
                           micro_ap_by_depth={f"{a}-{b}": micro_ap(Pg[:, (dg >= a) & (dg <= b)], Yg[:, (dg >= a) & (dg <= b)])
                                              for a, b in DEPTH_BINS})
        res["metrics"] = m
        # protein-level: does the C2 - isotonic loss track a covariate?
        d = ppa["C2_P7.3_w"] - ppa["isotonic"]
        cov = dict(cold_odds_mass=lam[idx], core_size=core[idx].sum(1).astype(float),
                   max_score=S[gr][:, cols].max(1), labels=Yg.sum(1).astype(float), diamond_hit=hit[gr].astype(float))
        res["per_protein_ap_C2_P7.3_minus_isotonic"] = dict(
            mean=float(np.nanmean(d)), q10_q50_q90=np.nanquantile(d, [0.1, 0.5, 0.9]).tolist(),
            spearman={k: float(spearmanr(v, d, nan_policy="omit")[0]) for k, v in cov.items()},
            by_diamond_hit={str(h): float(np.nanmean(d[hit[gr] == h])) for h in (False, True)})
        out[g] = res
    info["groups"] = out
    for g, res in out.items():
        print("==", g, json.dumps(res["covariates"]), flush=True)
        for name, mm in res["metrics"].items():
            print(f"  {name:12s} " + "  ".join(f"{a} {b:.4f}" for a, b in mm.items() if not isinstance(b, dict))
                  + "  depth " + json.dumps({a: round(b, 4) for a, b in mm["micro_ap_by_depth"].items()}), flush=True)
        print("  per-protein", json.dumps(res["per_protein_ap_C2_P7.3_minus_isotonic"]), flush=True)
    json.dump(info, open(RESULTS_DIR / f"p7_4_go_bp_c2_time_shift_diagnosis_{args.input}.json", "w"), indent=1)
    np.savez_compressed(DATA_DIR / "scores" / f"p7_4_predictions_{NS}_{args.input}.npz", rows=rows, core=core,
                        **{k.replace(".", "_"): v.astype(np.float32) for k, v in P.items()})


if __name__ == "__main__":
    main()
