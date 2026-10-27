"""P1.2b  Baseline: d4 on the FULL theory with zero-score terms passed as unit clauses (-y).

Purpose (Codex audit, 2026-10-19): in P1.2 every certified core equalled DiamondScore's
non-zero support, so "prune cold set + d4 on the core" may be nothing more than d4's own
unit propagation of the exact zeros. This runs that plain baseline on exactly the same
proteins (accessions taken from the P1.2 result file) with the same d4 time limit, and
checks on a few proteins that its marginals agree with the P1.2 certified marginals.

Output: results/p1_2b_d4_zero_units_<ontology>_<key>.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from wmc_common import DATA_DIR, RESULTS_DIR, load_theory
from wmc_inference import Circuit, certified_marginals, compile_with_d4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", default="go_bp")
    ap.add_argument("--diamond-key", default="prop")
    ap.add_argument("--clip", type=float, default=1e-9)
    ap.add_argument("--d4-timeout", type=float, default=600)
    ap.add_argument("--verify", type=int, default=5, help="proteins to cross-check against P1.2")
    ap.add_argument("--limit", type=int, default=0, help="smoke tests only")
    args = ap.parse_args()

    ids, edges = load_theory(args.ontology)
    n = len(ids)
    key = args.diamond_key
    ref = json.load(open(RESULTS_DIR / f"p1_2_certified_{args.ontology}_{key}_clip{args.clip:g}.json"))
    with np.load(DATA_DIR / "scores" / f"diamondscore_{args.ontology}.npz") as z:
        mat = csr_matrix((z[f"{key}_data"], z[f"{key}_indices"], z[f"{key}_indptr"]),
                         shape=(len(z["accessions"]), int(z["n_terms"])))
        row_of = {a: i for i, a in enumerate(z["accessions"].tolist())}
    proteins = ref["proteins"][: args.limit or None]

    recs = []
    with tempfile.TemporaryDirectory() as td:
        for j, pr in enumerate(proteins):
            raw = mat[row_of[pr["protein"]]].toarray().ravel().astype(float)
            zeros = np.flatnonzero(raw == 0)
            p = np.clip(raw, args.clip, 1 - args.clip)
            cnf, nnf = Path(td) / "full.cnf", Path(td) / "full.nnf"
            with open(cnf, "w") as fh:
                fh.write(f"p cnf {n} {len(edges) + len(zeros)}\n")
                for c, q in edges:
                    fh.write(f"-{c + 1} {q + 1} 0\n")
                for v in zeros:
                    fh.write(f"-{v + 1} 0\n")
            rec = dict(protein=pr["protein"], selection=pr["selection"], n_nonzero=int(len(raw) - len(zeros)),
                       p1_2_core_seconds=pr.get("core_seconds"), p1_2_failed="failed" in pr)
            t0 = time.time()
            try:
                compile_with_d4(cnf, nnf, timeout_s=args.d4_timeout)
                circ = Circuit(nnf, n)
                mu, _ = circ.marginals(p)
                rec["seconds"] = round(time.time() - t0, 3)
                rec["circuit_nodes"] = circ.n_nodes
                rec["zero_terms_nonzero_marginal"] = float(mu[zeros].max()) if len(zeros) else 0.0
                if j < args.verify and "failed" not in pr:
                    cert = certified_marginals(n, edges, p, 1e-3, Path(td), args.d4_timeout)  # own temp dir (was /tmp: collision risk, Codex audit)
                    rec["max_diff_vs_p1_2_certified"] = float(np.abs(mu - cert["mu_hat"]).max())
                    rec["p1_2_delta"] = cert["delta"]
            except Exception as exc:
                rec["failed"] = repr(exc)[:200]
                rec["seconds"] = round(time.time() - t0, 3)
            recs.append(rec)
            print(json.dumps(rec), flush=True)

    out = RESULTS_DIR / f"p1_2b_d4_zero_units_{args.ontology}_{key}.json"
    if not args.limit:
        with open(out, "w") as fh:
            json.dump(dict(args=vars(args), terms=n, proteins=recs), fh, indent=1)
        print(f"written {out}")


if __name__ == "__main__":
    main()
