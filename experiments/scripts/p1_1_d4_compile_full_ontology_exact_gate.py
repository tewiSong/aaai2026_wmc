"""P1.1  Gate test: can a strong exact solver (d4 decision-DNNF compiler) handle the full
ontology, and what does all-term inference cost once compiled?

Purpose: the paper's scale claim needs "exact inference is infeasible" to hold against a
strong solver, not only against dense JT. d4 exploits determinism and component caching,
which P0.4 cannot see. Outcomes:
  compiles     -> exact all-term ground truth for P2 on this ontology, and the
                  compile-once / evaluate-per-score-vector cost to beat
  fails        -> the hardness premise holds for d4 on this ontology (record time/memory)

Output: data/compiled/<ontology>.nnf (scratch) and results/p1_1_d4_<ontology>.json
"""
from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np

from wmc_common import DATA_DIR, RESULTS_DIR, THEORY_DIR, load_theory
from wmc_inference import Circuit, compile_with_d4, make_scores, violations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--timeout-hours", type=float, default=46)
    ap.add_argument("--eval-vectors", type=int, default=5)
    args = ap.parse_args()

    ids, edges = load_theory(args.ontology)
    n = len(ids)
    out_dir = DATA_DIR / "compiled"
    out_dir.mkdir(parents=True, exist_ok=True)
    nnf = out_dir / f"{args.ontology}.nnf"
    rec = dict(ontology=args.ontology, terms=n, clauses=int(len(edges)),
               timeout_hours=args.timeout_hours)
    try:
        rec["compile"] = compile_with_d4(THEORY_DIR / f"{args.ontology}.cnf", nnf,
                                         timeout_s=args.timeout_hours * 3600)
    except Exception as exc:
        rec["compile"] = dict(failed=repr(exc)[:500])
    rec["compile_peak_rss_gb_children"] = round(
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024 ** 2, 2)

    if "failed" not in rec["compile"]:
        rec["nnf_bytes"] = nnf.stat().st_size
        t0 = time.time()
        circ = Circuit(nnf, n)
        rec["circuit"] = dict(nodes=circ.n_nodes, edges=circ.n_edges,
                              parse_seconds=round(time.time() - t0, 2))
        rng = np.random.default_rng(0)
        evals = []
        for k in range(args.eval_vectors):
            fam = ["uniform", "beta22", "beta0505", "sparse"][k % 4]
            p = make_scores(n, fam, rng)
            t0 = time.time()
            mu, logz = circ.marginals(p)
            evals.append(dict(family=fam, seconds=round(time.time() - t0, 3), logZ=logz,
                              violations=violations(edges, mu)))
        rec["all_marginals_per_score_vector"] = evals
    print(json.dumps(rec, indent=2))
    with open(RESULTS_DIR / f"p1_1_d4_{args.ontology}.json", "w") as fh:
        json.dump(rec, fh, indent=2)


if __name__ == "__main__":
    main()
