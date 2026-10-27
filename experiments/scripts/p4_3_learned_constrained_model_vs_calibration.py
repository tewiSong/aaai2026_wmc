"""P4.3  Stage 1 (EXPLORATORY): does a learned constrained model C beat fair non-WMC methods?

Plan section 7.11. DeepGOPlus 1.0.6 test proteins with >= 1 non-root label are split 50/50
(seed 0) into a fit half and an evaluation half. Because the full test set was already analysed
in P4.0-P4.2, this split is exploration only, not independent validation.

Methods (all fitted on the fit half, evaluated on the evaluation half, root term excluded):
  raw, max_closure, isotonic                    no fitting
  exact_wmc_raw                                  plain conditioning (C with a=1, b=0)
  C                                              learned constrained model (WMC in the loop)
  depth_platt_iso / term_platt_iso / neighbor_lr_iso   non-WMC calibration (+ isotonic)
Metrics: Fmax, micro AP, Brier, per-label log-loss, ECE; 1,000 paired bootstrap resamples of
evaluation proteins for C minus every other method.

Output: results/p4_3_learned_C_<ns>_<input>.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.special import logit

from consistency import isotonic_dag_hildreth
from learned_constrained import (ModelC, apply_depth_platt, apply_neighbor_lr, apply_term_platt,
                                 fit_depth_platt, fit_neighbor_lr, fit_term_platt, neighbor_features,
                                 term_depth)
from p4_0_downstream_conditioning_deepgoplus_cc_mf import ece, fmax, max_closure
from p4_1b_downstream_consistency_methods_deepgoplus import micro_ap
from wmc_common import DATA_DIR, RESULTS_DIR, load_theory, write_dimacs
from wmc_inference import Circuit, compile_with_d4


def logloss(P, Y):
    q = np.clip(P, 1e-12, 1 - 1e-12)
    return float(-np.mean(Y * np.log(q) + (1 - Y) * np.log1p(-q)))


def metrics(P, Y):
    return dict(fmax=fmax(P, Y), micro_ap=micro_ap(P, Y), brier=float(np.mean((P - Y) ** 2)),
                logloss=logloss(P, Y), ece=ece(P, Y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True)
    ap.add_argument("--input", choices=["cnn", "fused"], required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--maxiter", type=int, default=200)
    args = ap.parse_args()

    ids, edges = load_theory(f"dgp2_{args.ns}")
    n = len(ids)
    z = np.load(DATA_DIR / "scores" / f"deepgoplus2_{args.ns}.npz")
    S, Y, root = z[args.input].astype(float), z["labels"].astype(np.int8), int(z["root_index"])
    cols = np.array([j for j in range(n) if j != root])
    keep = Y[:, cols].sum(1) > 0
    S, Y = S[keep], Y[keep]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(S))
    fit_idx, ev_idx = perm[: len(S) // 2], perm[len(S) // 2:]
    L = logit(np.clip(S, 1e-6, 1 - 1e-6))
    depth = term_depth(n, edges, root)

    with tempfile.TemporaryDirectory() as td:
        write_dimacs(Path(td) / "t.cnf", n, edges)
        compile_with_d4(Path(td) / "t.cnf", Path(td) / "t.nnf")
        circ = Circuit(Path(td) / "t.nnf", n)
    info = dict(theory=f"dgp2_{args.ns}", input=args.input, terms=n, proteins_fit=int(len(fit_idx)),
                proteins_eval=int(len(ev_idx)), max_depth=int(depth.max()), stage="exploratory split")

    model = ModelC(circ, depth, ridge=1e-3)
    t0 = time.time()
    res = model.fit(L[fit_idx], Y[fit_idx], maxiter=args.maxiter)
    info["C_fit"] = dict(seconds=round(time.time() - t0, 1), converged=bool(res.success), iterations=int(res.nit),
                         nll_fit=float(res.fun), a=model.params[: model.D].tolist(), b=model.params[model.D:].tolist())
    ident = ModelC(circ, depth); ident.params = np.r_[np.ones(model.D), np.zeros(model.D)]

    Lf, Yf, Le, Ye = L[fit_idx], Y[fit_idx], L[ev_idx], Y[ev_idx]
    Wd = fit_depth_platt(Lf, Yf, depth)
    Wt = fit_term_platt(Lf, Yf)
    Ff, Fe = neighbor_features(Lf, edges, n), neighbor_features(Le, edges, n)
    Wn = fit_neighbor_lr(Ff, Yf, depth)
    info["depth_platt"] = Wd.tolist()
    iso = lambda P: isotonic_dag_hildreth(P, edges)[0]
    Se = S[ev_idx]
    outs = {
        "raw": Se, "max_closure": max_closure(Se, edges, n), "isotonic": iso(Se),
        "exact_wmc_raw": ident.predict(Le), "C": model.predict(Le),
        "depth_platt_iso": iso(apply_depth_platt(Wd, Le, depth)),
        "term_platt_iso": iso(apply_term_platt(Wt, Le)),
        "neighbor_lr_iso": iso(apply_neighbor_lr(Wn, Fe, depth)),
    }
    ev = {k: v[:, cols] for k, v in outs.items()}
    Yc = Ye[:, cols]
    info["metrics"] = {k: metrics(v, Yc) for k, v in ev.items()}
    for k, m in info["metrics"].items():
        print(f"{k:16s} " + "  ".join(f"{a} {b:.5f}" for a, b in m.items()), flush=True)

    diffs = {k: {m: [] for m in ("fmax", "micro_ap", "brier", "logloss")} for k in ev if k != "C"}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(Yc), len(Yc))
        mc = metrics(ev["C"][idx], Yc[idx])
        for k in diffs:
            mk = metrics(ev[k][idx], Yc[idx])
            for m in diffs[k]:
                diffs[k][m].append(mc[m] - mk[m])
    info["C_minus_other_95ci"] = {k: {m: np.quantile(v, [0.025, 0.5, 0.975]).tolist() for m, v in d.items()}
                                  for k, d in diffs.items()}
    json.dump(info, open(RESULTS_DIR / f"p4_3_learned_C_{args.ns}_{args.input}.json", "w"), indent=1)
    print(json.dumps(info["C_minus_other_95ci"], indent=1))


if __name__ == "__main__":
    main()
