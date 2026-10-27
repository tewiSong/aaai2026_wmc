"""P4.6  Ablation for P4.5 (v2: rerun after the stable log-partition fix; adds theta*-feature
and whole-vector PCA baselines): is C2's gain due to joint constrained (WMC) modelling, or just to
its input features?

Same split (seed 0) and evaluation half as P4.3/P4.5. Non-WMC models with C2's exact feature set
[1, logit(s_i), logit(iso_i)] and per-depth parameters (3 per depth, like C2), fitted by
independent per-label log-loss on the fit half, then projected by isotonic regression:
  same_features_lr_iso       sigmoid(w_d . [1, l, m]) -> isotonic
  same_features_lr           same, without the final projection (may violate the hierarchy)
  same_plus_neighbors_lr_iso [1, l, m, max l_par, min l_par, max l_child] -> isotonic (6/depth)
Compared with C2 from results/p4_5_nested_C2_<ns>_<input>.json (re-predicted here from the
saved weights and cached theta*), with paired bootstrap of C2 minus each ablation.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from consistency import isotonic_dag_hildreth
from learned_constrained import ModelC2, _logloss_fit, neighbor_features, term_depth
from p4_3_learned_constrained_model_vs_calibration import metrics
from wmc_common import DATA_DIR, RESULTS_DIR, load_theory, write_dimacs
from wmc_inference import Circuit, compile_with_d4


def fit_depth_lr(F, Y, depth, ridge=1e-3):
    D, k = int(depth.max()) + 1, F.shape[-1]
    target = np.zeros(k); target[1] = 1.0
    W = np.tile(target, (D, 1))
    for d in range(D):
        m = depth == d
        if m.any():
            W[d] = _logloss_fit(F[:, m].reshape(-1, k), Y[:, m].ravel().astype(float), W[d], ridge, target)
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--bootstrap", type=int, default=1000)
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
    depth = term_depth(n, edges, root)
    with tempfile.TemporaryDirectory() as td:
        write_dimacs(Path(td) / "t.cnf", n, edges); compile_with_d4(Path(td) / "t.cnf", Path(td) / "t.nnf")
        circ = Circuit(Path(td) / "t.nnf", n)
    mu_unif, _ = circ.marginals(np.full(n, 0.5))
    ISO = isotonic_dag_hildreth(S, edges)[0]
    MU_T = (1 - args.eps) * ISO + args.eps * mu_unif
    L, M = logit(S), logit(np.clip(MU_T, 1e-9, 1 - 1e-9))

    p45 = json.load(open(RESULTS_DIR / f"p4_5_nested_C2_{args.ns}_{args.input}.json"))
    TH0 = np.load(DATA_DIR / "scores" / f"theta_star_{args.ns}_{args.input}_eps{args.eps:g}_v2.npz")["theta"]
    c2 = ModelC2(circ, depth); c2.w = np.array(p45["C2_fit"]["w"]).ravel()

    F = np.stack([np.ones_like(L), L, M], -1)
    NB = neighbor_features(L, edges, n)[..., 2:]                   # max par, min par, max child
    F2 = np.concatenate([F, NB], -1)
    W1 = fit_depth_lr(F[fit_idx], Y[fit_idx], depth)
    W2 = fit_depth_lr(F2[fit_idx], Y[fit_idx], depth)
    # (Codex audit) theta* as an input feature, but independent prediction: separates the
    # whole-vector parameter transform (needs WMC once) from joint prediction under constraints
    F3 = np.concatenate([F, TH0[..., None] / 10.0], -1)
    W3 = fit_depth_lr(F3[fit_idx], Y[fit_idx], depth)
    # (plan 7.14 a) whole-vector non-WMC baseline: top-32 principal components of the protein's
    # full logit vector (fitted on the fit half) appended to every term's features
    Lc = L - L[fit_idx].mean(0)
    _, _, Vt = np.linalg.svd(Lc[fit_idx], full_matrices=False)
    Z = Lc @ Vt[:32].T / np.sqrt(len(fit_idx))
    F4 = np.concatenate([F, np.repeat(Z[:, None, :], n, 1)], -1)
    W4 = fit_depth_lr(F4[fit_idx], Y[fit_idx], depth)
    pred = lambda W, X: expit(np.einsum("kni,ni->kn", X, W[depth]))
    iso = lambda P: isotonic_dag_hildreth(P, edges)[0]
    outs = {
        "C2": c2.predict(TH0[ev_idx], L[ev_idx], M[ev_idx]),
        "same_features_lr_iso": iso(pred(W1, F[ev_idx])),
        "same_features_lr": pred(W1, F[ev_idx]),
        "same_plus_neighbors_lr_iso": iso(pred(W2, F2[ev_idx])),
        "theta_star_feature_lr_iso": iso(pred(W3, F3[ev_idx])),
        "whole_vector_pca32_lr_iso": iso(pred(W4, F4[ev_idx])),
        "isotonic": ISO[ev_idx],
    }
    ev = {k: v[:, cols] for k, v in outs.items()}
    Yc = Y[ev_idx][:, cols]
    res = dict(theory=f"dgp2_{args.ns}", input=args.input, proteins_eval=int(len(ev_idx)),
               metrics={k: metrics(v, Yc) for k, v in ev.items()})
    for k, m in res["metrics"].items():
        print(f"{k:28s} " + "  ".join(f"{a} {b:.5f}" for a, b in m.items()), flush=True)
    diffs = {k: {m: [] for m in ("fmax", "micro_ap", "brier", "logloss")} for k in ev if k != "C2"}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(Yc), len(Yc))
        mc = metrics(ev["C2"][idx], Yc[idx])
        for k in diffs:
            mk = metrics(ev[k][idx], Yc[idx])
            for m in diffs[k]:
                diffs[k][m].append(mc[m] - mk[m])
    res["C2_minus_other_95ci"] = {k: {m: np.quantile(v, [0.025, 0.5, 0.975]).tolist() for m, v in d.items()}
                                  for k, d in diffs.items()}
    json.dump(res, open(RESULTS_DIR / f"p4_6_ablation_{args.ns}_{args.input}.json", "w"), indent=1)
    print(json.dumps(res["C2_minus_other_95ci"], indent=1))


if __name__ == "__main__":
    main()
