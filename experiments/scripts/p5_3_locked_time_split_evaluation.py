"""P5.3  Locked time-split evaluation of C2 (protocol P5_protocol_time_split_confirmation.md).

Refuses to run if the protocol file's sha256 differs from results/p5_protocol_lock.txt.
Fits every method on ALL old proteins of the namespace (DeepGOPlus 1.0.6 test proteins with a
non-root label; theta* from the P4.5 v2 cache, same eps / reference), evaluates on the P5
targets (P5.1 labels, P5.2 scores), and applies the pre-registered decision rule.
Output: results/p5_3_eval_<ns>_<input>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import logit

from consistency import isotonic_dag_hildreth
from learned_constrained import (ModelC2, apply_depth_platt, apply_neighbor_lr, apply_term_platt, fit_depth_platt,
                                 fit_neighbor_lr, fit_term_platt, moment_match, neighbor_features, term_depth)
from p4_0_downstream_conditioning_deepgoplus_cc_mf import max_closure
from p4_3_learned_constrained_model_vs_calibration import metrics
from p4_6_ablation_same_features_without_wmc import fit_depth_lr
from scipy.special import expit
from wmc_common import DATA_DIR, EXP_DIR, RESULTS_DIR, load_theory, write_dimacs
from wmc_inference import Circuit, compile_with_d4

_C = None


def _init(nnf, n):
    global _C
    _C = Circuit(Path(nnf), n)


def _mm(mu_t):
    return moment_match(_C, mu_t, logit(np.clip(mu_t, 1e-6, 1 - 1e-6)))


def check_lock():
    lock = open(RESULTS_DIR / "p5_protocol_lock.txt").read().split()[2]
    now = hashlib.sha256(open(EXP_DIR / "P5_protocol_time_split_confirmation.md", "rb").read()).hexdigest()
    if lock != now:
        raise SystemExit(f"protocol changed after locking ({now} != {lock}); record an amendment first")
    return now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True)
    ap.add_argument("--input", choices=["cnn", "fused"], required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args = ap.parse_args()
    lock = check_lock()
    eps = 1e-3

    ids, edges = load_theory(f"dgp2_{args.ns}")
    n = len(ids)
    old = np.load(DATA_DIR / "scores" / f"deepgoplus2_{args.ns}.npz")
    S0, Y0, root = old[args.input].astype(float), old["labels"].astype(np.int8), int(old["root_index"])
    cols = np.array([j for j in range(n) if j != root])
    keep = Y0[:, cols].sum(1) > 0
    S0, Y0 = np.clip(S0[keep], 1e-6, 1 - 1e-6), Y0[keep]
    new = np.load(DATA_DIR / "scores" / f"p5_scores_{args.ns}.npz")
    if args.input == "fused" and not bool(new["fused_valid"]):
        raise SystemExit("gate (b) failed for this namespace: fused input is not part of the confirmation")
    S1, Y1 = np.clip(new[args.input].astype(float), 1e-6, 1 - 1e-6), new["labels"].astype(np.int8)
    assert int(new["root_index"]) == root
    depth = term_depth(n, edges, root)

    td = tempfile.mkdtemp()
    write_dimacs(Path(td) / "t.cnf", n, edges); compile_with_d4(Path(td) / "t.cnf", Path(td) / "t.nnf")
    circ = Circuit(Path(td) / "t.nnf", n)
    mu_unif, _ = circ.marginals(np.full(n, 0.5))
    iso = lambda P: isotonic_dag_hildreth(P, edges)[0]
    ISO0, ISO1 = iso(S0), iso(S1)
    MT0, MT1 = (1 - eps) * ISO0 + eps * mu_unif, (1 - eps) * ISO1 + eps * mu_unif
    TH0 = np.load(DATA_DIR / "scores" / f"theta_star_{args.ns}_{args.input}_eps0.001_v2.npz")["theta"]
    with Pool(args.workers, initializer=_init, initargs=(str(Path(td) / "t.nnf"), n)) as pool:
        res = list(pool.imap(_mm, list(MT1), chunksize=4))
    TH1 = np.stack([r[0] for r in res]); mm_err = np.array([r[1] for r in res])
    L0, L1 = logit(S0), logit(S1)
    M0, M1 = logit(np.clip(MT0, 1e-9, 1 - 1e-9)), logit(np.clip(MT1, 1e-9, 1 - 1e-9))

    info = dict(protocol_sha256=lock, theory=f"dgp2_{args.ns}", input=args.input, fit_proteins=int(len(S0)),
                eval_targets=int(len(S1)), target_moment_match=dict(err_median=float(np.median(mm_err)),
                                                                 err_max=float(mm_err.max())))
    c2 = ModelC2(circ, depth, ridge=1e-3)
    r = c2.fit(TH0, L0, M0, Y0, maxiter=1000)
    info["C2_fit"] = dict(converged=bool(r.success), iterations=int(r.nit), nll=float(r.fun))

    Wd, Wt = fit_depth_platt(L0, Y0, depth), fit_term_platt(L0, Y0)
    Wn = fit_neighbor_lr(neighbor_features(L0, edges, n), Y0, depth)
    F0, F1 = np.stack([np.ones_like(L0), L0, M0], -1), np.stack([np.ones_like(L1), L1, M1], -1)
    W1 = fit_depth_lr(F0, Y0, depth)
    G0, G1 = np.concatenate([F0, TH0[..., None] / 10.0], -1), np.concatenate([F1, TH1[..., None] / 10.0], -1)
    W3 = fit_depth_lr(G0, Y0, depth)
    mu_l = L0.mean(0)
    _, _, Vt = np.linalg.svd(L0 - mu_l, full_matrices=False)
    Z0, Z1 = (L0 - mu_l) @ Vt[:32].T / np.sqrt(len(L0)), (L1 - mu_l) @ Vt[:32].T / np.sqrt(len(L0))
    H0 = np.concatenate([F0, np.repeat(Z0[:, None, :], n, 1)], -1)
    H1 = np.concatenate([F1, np.repeat(Z1[:, None, :], n, 1)], -1)
    W4 = fit_depth_lr(H0, Y0, depth)
    pred = lambda W, X: expit(np.einsum("kni,ni->kn", X, W[depth]))

    outs = {
        "raw": S1, "max_closure": max_closure(S1, edges, n), "isotonic": ISO1,
        "depth_platt_iso": iso(apply_depth_platt(Wd, L1, depth)),
        "term_platt_iso": iso(apply_term_platt(Wt, L1)),
        "neighbor_lr_iso": iso(apply_neighbor_lr(Wn, neighbor_features(L1, edges, n), depth)),
        "same_features_lr_iso": iso(pred(W1, F1)),
        "theta_star_feature_lr_iso": iso(pred(W3, G1)),
        "whole_vector_pca32_lr_iso": iso(pred(W4, H1)),
        "C2": c2.predict(TH1, L1, M1),
    }
    ev = {k: v[:, cols] for k, v in outs.items()}
    Yc = Y1[:, cols]
    info["metrics"] = {k: metrics(v, Yc) for k, v in ev.items()}
    for k, m in info["metrics"].items():
        print(f"{k:28s} " + "  ".join(f"{a} {b:.5f}" for a, b in m.items()), flush=True)
    rng = np.random.default_rng(0)
    diffs = {k: {m: [] for m in ("fmax", "micro_ap", "brier", "logloss", "ece")} for k in ev if k != "C2"}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(Yc), len(Yc))
        mc = metrics(ev["C2"][idx], Yc[idx])
        for k in diffs:
            mk = metrics(ev[k][idx], Yc[idx])
            for m in diffs[k]:
                diffs[k][m].append(mc[m] - mk[m])
    ci = {k: {m: np.quantile(v, [0.025, 0.5, 0.975]).tolist() for m, v in d.items()} for k, d in diffs.items()}
    info["C2_minus_other_95ci"] = ci
    pos = lambda k: ci[k]["fmax"][0] > 0 and ci[k]["micro_ap"][0] > 0
    fitted = ["depth_platt_iso", "term_platt_iso", "neighbor_lr_iso", "same_features_lr_iso",
              "theta_star_feature_lr_iso", "whole_vector_pca32_lr_iso"]
    info["H1_C2_beats_isotonic"] = bool(pos("isotonic"))
    info["H2_C2_beats_every_fitted_baseline"] = bool(all(pos(k) for k in fitted))
    info["H2_detail"] = {k: bool(pos(k)) for k in fitted}
    json.dump(info, open(RESULTS_DIR / f"p5_3_eval_{args.ns}_{args.input}.json", "w"), indent=1)
    print(json.dumps({k: info[k] for k in ("H1_C2_beats_isotonic", "H2_C2_beats_every_fitted_baseline", "H2_detail")}, indent=1))


if __name__ == "__main__":
    main()
