"""P1.4a  Dense neural scores: exact all-term marginals by compiling the vocabulary theory ONCE,
and BP's error against them for every DeepGOPlus test protein.

Purpose: decide whether dense real scores (DeepGOPlus CNN, every term non-zero) need a
dedicated method. Where d4 compiles the theory, compile-once/evaluate-per-protein is the
exact baseline any new method must beat; BP's error measures what heuristics miss. Where d4
cannot compile (GO bp vocabulary, 4,474 terms), that failure itself is the evidence that a
scalable method with guarantees is needed (P1.4b provides a Gibbs-based reference there).

Output: data/compiled/dgp_<ns>.nnf; results/p1_4a_dgp_<ns>.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from wmc_common import DATA_DIR, RESULTS_DIR, THEORY_DIR, load_theory
from wmc_inference import Circuit, compile_with_d4, loopy_bp, violations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True)
    ap.add_argument("--compile-timeout-hours", type=float, default=46)
    ap.add_argument("--max-proteins", type=int, default=0)
    args = ap.parse_args()

    name = f"dgp_{args.ns}"
    ids, edges = load_theory(name)
    n = len(ids)
    z = np.load(DATA_DIR / "scores" / f"deepgoplus_{args.ns}.npz")
    S, accs = z["scores"].astype(np.float64), z["accessions"]
    if args.max_proteins:
        S, accs = S[: args.max_proteins], accs[: args.max_proteins]
    nnf = DATA_DIR / "compiled" / f"{name}.nnf"
    nnf.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(theory=name, terms=n, edges=int(len(edges)), proteins=int(len(S)))
    try:
        rec["compile"] = compile_with_d4(THEORY_DIR / f"{name}.cnf", nnf,
                                         timeout_s=args.compile_timeout_hours * 3600)
    except Exception as exc:
        rec["compile"] = dict(failed=repr(exc)[:300])
        print(json.dumps(rec), flush=True)
        json.dump(rec, open(RESULTS_DIR / f"p1_4a_{name}.json", "w"), indent=1)
        return
    t0 = time.time()
    circ = Circuit(nnf, n)
    rec["circuit"] = dict(nodes=circ.n_nodes, edges=circ.n_edges, parse_seconds=round(time.time() - t0, 2))

    rows = []
    for i in range(len(S)):
        p = S[i]
        t0 = time.time(); mu, _ = circ.marginals(p); te = time.time() - t0
        t0 = time.time(); bp = loopy_bp(n, edges, p); tb = time.time() - t0
        err = np.abs(bp["mu"] - mu)
        rows.append(dict(protein=str(accs[i]), exact_seconds=round(te, 4), bp_seconds=round(tb, 3),
                         bp_converged=bp["converged"], bp_max_abs_err=float(err.max()),
                         bp_mae=float(err.mean()), bp_terms_err_gt_1e2=int((err > 1e-2).sum()),
                         bp_violations=violations(edges, bp["mu"]),
                         n_intermediate=int(((p > 0.05) & (p < 0.95)).sum()),
                         max_shift_from_scores=float(np.abs(mu - p).max())))
        if i % 500 == 0:
            print(i, json.dumps(rows[-1]), flush=True)
    mx = np.array([r["bp_max_abs_err"] for r in rows])
    rec["summary"] = dict(
        bp_max_err_quantiles=np.quantile(mx, [0.5, 0.9, 0.99, 1.0]).tolist(),
        frac_proteins_bp_err_gt_1e3=float(np.mean(mx > 1e-3)), frac_proteins_bp_err_gt_1e2=float(np.mean(mx > 1e-2)),
        exact_seconds_median=float(np.median([r["exact_seconds"] for r in rows])),
        bp_seconds_median=float(np.median([r["bp_seconds"] for r in rows])),
        conditioning_shift_quantiles=np.quantile([r["max_shift_from_scores"] for r in rows], [0.5, 0.9, 1.0]).tolist())
    rec["per_protein"] = rows
    print(json.dumps(rec["summary"], indent=1))
    json.dump(rec, open(RESULTS_DIR / f"p1_4a_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
