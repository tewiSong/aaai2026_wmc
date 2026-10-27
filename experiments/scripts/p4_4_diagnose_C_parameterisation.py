"""P4.4  Why did the learned constrained model C fail? Diagnose its parameterisation.

Hypothesis: the constrained product family {P_theta(y) ~ exp(theta.y)[y valid]} can reach any
interior consistent marginal vector, but C's map theta_i = a_d logit(s_i) + b_d cannot express
the parameters that reproduce good marginals, so C is not nested with the baselines it was
compared against.

For a sample of proteins:
 1. target mu* = isotonic(s), moved slightly into the interior of the order polytope:
    mu* = (1-eps) iso(s) + eps * mu_uniform (uniform distribution over valid assignments);
 2. moment matching: theta* = argmin log Z(theta) - theta . mu*  (convex; gradient mu(theta)-mu*),
    marginals by the exact d4 circuit; report achieved max |mu(theta*) - mu*|;
 3. how well can theta* be explained by (a) a per-depth affine function of logit(s_i) [= C's
    family], (b) adding parent/child score features [tree reparameterisation form
    logit(s_i/s_par) + sum_children log(1 - s_j/s_i)]? Report R^2 per feature set and the
    downstream metrics of the marginals each restricted fit induces.
Output: results/p4_4_C_parameterisation_<ns>_<input>.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from consistency import isotonic_dag_hildreth
from learned_constrained import term_depth
from wmc_common import DATA_DIR, RESULTS_DIR, load_theory, write_dimacs
from wmc_inference import Circuit, compile_with_d4


def log_partition(circ, theta):
    p = np.clip(expit(theta), 1e-12, 1 - 1e-12)
    mu, logz_c = circ.marginals(p)
    return logz_c + np.sum(np.logaddexp(0.0, theta)), mu   # log sum_valid exp(theta.y)


def moment_match(circ, mu_t, theta0, maxiter=2000):
    def f(th):
        lz, mu = log_partition(circ, th)
        return lz - th @ mu_t, mu - mu_t
    res = minimize(f, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, gtol=1e-10))
    _, mu = log_partition(circ, res.x)
    return res.x, float(np.abs(mu - mu_t).max()), int(res.nit)


def features(L, S, edges, n, kind):
    parents = [[] for _ in range(n)]
    children = [[] for _ in range(n)]
    for c, q in edges:
        parents[c].append(int(q)); children[q].append(int(c))
    cols = [np.ones(n), L]
    if kind == "tree":
        smin_par = np.array([S[parents[i]].min() if parents[i] else 1.0 for i in range(n)])
        ratio = np.clip(S / np.maximum(smin_par, 1e-6), 1e-6, 1 - 1e-6)
        child_term = np.array([np.sum(np.log1p(-np.clip(S[children[i]] / max(S[i], 1e-6), 0, 1 - 1e-6)))
                               if children[i] else 0.0 for i in range(n)])
        cols += [logit(ratio), child_term]
    return np.stack(cols, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="go_mf")
    ap.add_argument("--input", default="cnn")
    ap.add_argument("--proteins", type=int, default=100)
    ap.add_argument("--eps", type=float, default=1e-3)
    args = ap.parse_args()
    ids, edges = load_theory(f"dgp2_{args.ns}")
    n = len(ids)
    z = np.load(DATA_DIR / "scores" / f"deepgoplus2_{args.ns}.npz")
    S, Y, root = z[args.input].astype(float), z["labels"], int(z["root_index"])
    keep = np.flatnonzero(np.delete(Y, root, 1).sum(1) > 0)
    rng = np.random.default_rng(0)
    rows = rng.choice(keep, args.proteins, replace=False)
    depth = term_depth(n, edges, root)
    with tempfile.TemporaryDirectory() as td:
        write_dimacs(Path(td) / "t.cnf", n, edges); compile_with_d4(Path(td) / "t.cnf", Path(td) / "t.nnf")
        circ = Circuit(Path(td) / "t.nnf", n)
    mu_unif, _ = circ.marginals(np.full(n, 0.5))

    TH, LL, SS, MU, fails = [], [], [], [], []
    for r in rows:
        s = np.clip(S[r], 1e-6, 1 - 1e-6)
        iso = isotonic_dag_hildreth(s[None], edges)[0][0]
        mu_t = (1 - args.eps) * iso + args.eps * mu_unif
        th, err, it = moment_match(circ, mu_t, logit(np.clip(mu_t, 1e-6, 1 - 1e-6)))
        fails.append(err)
        TH.append(th); LL.append(logit(s)); SS.append(s); MU.append(mu_t)
    TH, LL, SS, MU = map(np.array, (TH, LL, SS, MU))
    out = dict(theory=f"dgp2_{args.ns}", input=args.input, proteins=len(rows), eps=args.eps,
               moment_match_max_abs_err=dict(median=float(np.median(fails)), max=float(np.max(fails))))

    for kind in ("depth_affine_own_score", "tree"):
        X = np.stack([features(LL[k], SS[k], edges, n, "tree" if kind == "tree" else "own") for k in range(len(rows))])
        pred = np.zeros_like(TH)
        for d in range(int(depth.max()) + 1):
            m = depth == d
            if not m.any():
                continue
            A = X[:, m].reshape(-1, X.shape[-1]); t = TH[:, m].ravel()
            w, *_ = np.linalg.lstsq(A, t, rcond=None)
            pred[:, m] = (X[:, m] @ w)
        ss_res = float(np.sum((TH - pred) ** 2)); ss_tot = float(np.sum((TH - TH.mean()) ** 2))
        mu_fit = np.stack([circ.marginals(np.clip(expit(pred[k]), 1e-12, 1 - 1e-12))[0] for k in range(len(rows))])
        out[kind] = dict(R2=1 - ss_res / ss_tot,
                         marginal_err_vs_target_median=float(np.median(np.abs(mu_fit - MU).max(1))),
                         marginal_mae_vs_target=float(np.abs(mu_fit - MU).mean()))
    # how different is theta* from the naive "theta = logit(s)" used by plain conditioning?
    out["theta_star_minus_logit_s_by_depth"] = [
        dict(depth=d, mean=float((TH - LL)[:, depth == d].mean()), sd=float((TH - LL)[:, depth == d].std()))
        for d in range(int(depth.max()) + 1) if (depth == d).any()]
    print(json.dumps(out, indent=1))
    json.dump(out, open(RESULTS_DIR / f"p4_4_C_parameterisation_{args.ns}_{args.input}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
