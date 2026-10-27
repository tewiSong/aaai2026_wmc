"""P1.4b  GO bp vocabulary (4,474 terms) under dense DeepGOPlus scores: BP against a
monotone-bracketed Gibbs reference.

Purpose: d4 does not compile this theory quickly (P1.4a tries for up to 46 h), so no exact
reference is available yet. For y_c <= y_p with product weights, heat-bath Gibbs is monotone;
chains started from all-zeros and all-ones bracket the stationary marginals. Where the bracket
has closed and the Monte Carlo error is small, |BP - Gibbs| well above that error is evidence
of BP error at full vocabulary scale. This is a statistical reference, not a certificate.

Output: results/p1_4b_dgp_go_bp_part<k>.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from wmc_common import DATA_DIR, RESULTS_DIR, load_theory
from wmc_inference import gibbs_marginals, loopy_bp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--proteins", type=int, default=100)
    ap.add_argument("--sweeps", type=int, default=10000)
    ap.add_argument("--burn-in", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=32)
    args = ap.parse_args()
    ids, edges = load_theory("dgp_go_bp")
    n = len(ids)
    z = np.load(DATA_DIR / "scores" / "deepgoplus_go_bp.npz")
    S, accs = z["scores"].astype(np.float64), z["accessions"]
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(S), args.proteins, replace=False)          # random proteins
    mine = chosen[args.part::args.parts]
    rows = []
    for i in mine:
        p = S[i]
        t0 = time.time(); bp = loopy_bp(n, edges, p); tb = time.time() - t0
        t0 = time.time()
        g = gibbs_marginals(n, edges, p, np.random.default_rng(int(i)), chains=args.chains,
                            burn_in=args.burn_in, sweeps=args.sweeps)
        tg = time.time() - t0
        tol = 4 * g["chain_se"] + g["bracket_gap"]           # per-term uncertainty of the reference
        d = np.abs(bp["mu"] - g["mu"])
        excess = np.maximum(d - tol, 0.0)
        rows.append(dict(protein=str(accs[i]), bp_seconds=round(tb, 2), gibbs_seconds=round(tg, 1),
                         bp_converged=bp["converged"], max_bracket_gap=float(g["bracket_gap"].max()),
                         max_chain_se=float(g["chain_se"].max()), max_abs_bp_minus_gibbs=float(d.max()),
                         max_excess_over_reference_uncertainty=float(excess.max()),
                         terms_excess_gt_1e2=int((excess > 1e-2).sum()),
                         n_intermediate=int(((p > 0.05) & (p < 0.95)).sum())))
        print(json.dumps(rows[-1]), flush=True)
    json.dump(dict(args=vars(args), terms=n, proteins=rows),
              open(RESULTS_DIR / f"p1_4b_dgp_go_bp_part{args.part}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
