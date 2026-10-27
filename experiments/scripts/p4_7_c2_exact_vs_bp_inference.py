"""P4.7  Does C2 need exact inference? C2 with exact circuit marginals vs C2 with loopy BP.

Plan 7.14 (b): if BP-based C2 is as good, exact / scalable-exact inference adds little; if it is
clearly worse, better scalable inference is the computational bottleneck at GO bp scale.
Both variants use the same Adam trainer (gradient = y - marginals), steps and learning rate, so
the only difference is the inference engine:
  C2_exact_adam  theta* by exact moment matching (cached, P4.5 v2), exact marginals in training/prediction
  C2_bp          given the EXACT theta* (favourable to BP: P4.7a-c showed BP/Bethe moment matching is
                 numerically ill-posed or wrong on these theories), BP marginals in training and prediction
  C2_bp_train_exact_pred   BP-trained weights, exact marginals at prediction
Also reports the L-BFGS C2 from P4.5, isotonic, and BP fidelity |mu_BP - mu_exact| at C2_bp's
parameters. Same exploratory split (seed 0) as P4.3 / P4.5.
Output: results/p4_7_c2_exact_vs_bp_<ns>_<input>.json
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
from learned_constrained import ModelC2, log_partition, term_depth, train_c2_adam
from p4_3_learned_constrained_model_vs_calibration import metrics
from wmc_common import DATA_DIR, RESULTS_DIR, load_theory, write_dimacs
from wmc_inference import Circuit, compile_with_d4, loopy_bp_theta

_E = _N = None


def _init(edges, n):
    global _E, _N
    _E, _N = edges, n


def _mmbp(mu_t):
    th, err, k = moment_match_bp_fast(_N, _E, mu_t)
    return th, err, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--limit", type=int, default=0)
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
    write_dimacs(Path(td) / "t.cnf", n, edges); compile_with_d4(Path(td) / "t.cnf", Path(td) / "t.nnf")
    circ = Circuit(Path(td) / "t.nnf", n)
    mu_unif, _ = circ.marginals(np.full(n, 0.5))
    ISO = np.zeros_like(S); ISO[use] = isotonic_dag_hildreth(S[use], edges)[0]
    MU_T = (1 - args.eps) * ISO + args.eps * mu_unif
    L, M = logit(S), logit(np.clip(MU_T, 1e-9, 1 - 1e-9))
    TH_EX = np.load(DATA_DIR / "scores" / f"theta_star_{args.ns}_{args.input}_eps{args.eps:g}_v2.npz")["theta"]

    TH_BP = TH_EX                      # BP variant receives the exact theta* (see docstring)
    info = dict(theory=f"dgp2_{args.ns}", input=args.input, stage="exploratory split", steps=args.steps, lr=args.lr,
                proteins_fit=int(len(fit_idx)), proteins_eval=int(len(ev_idx)), bp_variant="exact theta*, BP training+prediction")
    print(json.dumps(info), flush=True)

    exact = lambda th: log_partition(circ, th)[1]
    msgs = {}

    def bp_marg(key, th):
        r = loopy_bp_theta(n, edges, th, init=msgs.get(key), tol=1e-10, max_iter=5000)
        msgs[key] = r["messages"]
        return r["mu"]

    fi = fit_idx
    t0 = time.time()
    w_ex, h_ex = train_c2_adam(lambda k, th: exact(th), TH_EX[fi], L[fi], M[fi], Y[fi], depth, args.steps, args.lr)
    t_ex = time.time() - t0
    t0 = time.time()
    w_bp, h_bp = train_c2_adam(lambda k, th: bp_marg(("fit", k), th), TH_BP[fi], L[fi], M[fi], Y[fi], depth, args.steps, args.lr)
    t_bp = time.time() - t0
    info["training"] = dict(exact_seconds=round(t_ex, 1), bp_seconds=round(t_bp, 1), exact_grad_norm=h_ex[-1:],
                            bp_grad_norm=h_bp[-1:], w_exact=w_ex.round(4).tolist(), w_bp=w_bp.round(4).tolist())

    D, K = int(depth.max()) + 1, ModelC2.K
    theta = lambda w, TH, k: TH[k] + w.reshape(D, K)[depth][:, 0] + w.reshape(D, K)[depth][:, 1] * L[k] + w.reshape(D, K)[depth][:, 2] * M[k]
    ev = list(ev_idx)
    P_ex = np.stack([exact(theta(w_ex, TH_EX, k)) for k in ev])
    P_bp = np.stack([bp_marg(("ev", k), theta(w_bp, TH_BP, k)) for k in ev])
    P_bp_ex = np.stack([exact(theta(w_bp, TH_BP, k)) for k in ev])
    fidelity = np.abs(P_bp - P_bp_ex).max(1)
    info["bp_fidelity_at_C2_bp_params"] = dict(median=float(np.median(fidelity)), p90=float(np.quantile(fidelity, 0.9)),
                                               max=float(fidelity.max()))
    outs = {"isotonic": ISO[ev_idx], "C2_exact_adam": P_ex, "C2_bp": P_bp, "C2_bp_train_exact_pred": P_bp_ex}
    p45 = RESULTS_DIR / f"p4_5_nested_C2_{args.ns}_{args.input}.json"
    if p45.exists():
        c2 = ModelC2(circ, depth); c2.w = np.array(json.load(open(p45))["C2_fit"]["w"]).ravel()
        outs["C2_lbfgs_p45"] = c2.predict(TH_EX[ev_idx], L[ev_idx], M[ev_idx])
    evm = {k: v[:, cols] for k, v in outs.items()}
    Yc = Y[ev_idx][:, cols]
    info["metrics"] = {k: metrics(v, Yc) for k, v in evm.items()}
    for k, m in info["metrics"].items():
        print(f"{k:24s} " + "  ".join(f"{a} {b:.5f}" for a, b in m.items()), flush=True)
    pairs = [("C2_bp", "C2_exact_adam"), ("C2_bp", "isotonic"), ("C2_exact_adam", "isotonic"),
             ("C2_bp_train_exact_pred", "C2_bp"), ("C2_exact_adam", "C2_lbfgs_p45")]
    pairs = [p for p in pairs if p[0] in evm and p[1] in evm]
    diffs = {f"{a}-{b}": {m: [] for m in ("fmax", "micro_ap", "brier", "logloss")} for a, b in pairs}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(Yc), len(Yc))
        mm = {k: metrics(v[idx], Yc[idx]) for k, v in evm.items()}
        for a, b in pairs:
            for m in diffs[f"{a}-{b}"]:
                diffs[f"{a}-{b}"][m].append(mm[a][m] - mm[b][m])
    info["diff_95ci"] = {k: {m: np.quantile(v, [0.025, 0.5, 0.975]).tolist() for m, v in d.items()} for k, d in diffs.items()}
    if not args.limit:
        json.dump(info, open(RESULTS_DIR / f"p4_7_c2_exact_vs_bp_{args.ns}_{args.input}.json", "w"), indent=1)
    print(json.dumps(info["diff_95ci"], indent=1))


if __name__ == "__main__":
    main()
