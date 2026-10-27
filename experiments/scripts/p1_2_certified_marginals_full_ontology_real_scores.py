"""P1.2  Certified all-term marginals on a FULL ontology under real predictor scores, and how
far loopy BP is from them.

Purpose (H1 vs H2, plan section 7): P1.0a measures BP error only on <= 2,000-term subgraphs.
Here the reference covers the complete ontology (GO bp: 28k terms, where d4 cannot compile the
full theory): prune a down-closed cold set D with certificate delta = 1 - prod_D (1 - p_d),
solve the remaining active core exactly with d4, and obtain an interval for every term that
provably contains the true marginal. BP outside its interval is certifiably wrong by at least
the distance to the interval; BP inside means its error is at most the interval width.
The core sizes also test H2's method claim directly: are real posteriors confined to small
active cores?

Output: results/p1_2_certified_<ontology>_<key>_clip<clip>.json
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
from wmc_inference import certified_marginals, loopy_bp, violations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", default="go_bp")
    ap.add_argument("--diamond-key", default="prop")
    ap.add_argument("--proteins", type=int, default=100)
    ap.add_argument("--clip", type=float, default=1e-9)
    ap.add_argument("--delta-target", type=float, default=1e-3)
    ap.add_argument("--d4-timeout", type=float, default=1800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    ids, edges = load_theory(args.ontology)
    n = len(ids)
    key = args.diamond_key
    with np.load(DATA_DIR / "scores" / f"diamondscore_{args.ontology}.npz") as z:
        mat = csr_matrix((z[f"{key}_data"], z[f"{key}_indices"], z[f"{key}_indptr"]),
                         shape=(len(z["accessions"]), int(z["n_terms"])))
        accs = z["accessions"]
    inter = np.array([np.sum((mat[r].data > 0.05) & (mat[r].data < 0.95)) for r in range(mat.shape[0])])
    k = min(args.proteins, mat.shape[0])
    top = np.argsort(-inter, kind="stable")[: k // 2]
    rest = np.setdiff1d(np.arange(mat.shape[0]), top)
    rows = np.concatenate([top, rng.choice(rest, k - len(top), replace=False)]).astype(int)

    recs = []
    with tempfile.TemporaryDirectory() as td:
        for j, r in enumerate(rows):
            p = np.clip(mat[r].toarray().ravel().astype(float), args.clip, 1 - args.clip)
            rec = dict(protein=str(accs[r]), selection="most_intermediate" if j < len(top) else "random",
                       n_intermediate=int(inter[r]), n_nonzero=int(mat[r].nnz))
            try:
                cert = certified_marginals(n, edges, p, args.delta_target, Path(td), args.d4_timeout)
            except Exception as exc:
                rec["failed"] = repr(exc)[:300]
                recs.append(rec)
                print(json.dumps(rec), flush=True)
                continue
            t0 = time.time()
            bp = loopy_bp(n, edges, p)
            t_bp = time.time() - t0
            mu_bp = bp["mu"]
            outside = np.maximum(cert["lo"] - mu_bp, mu_bp - cert["hi"])
            width = cert["hi"] - cert["lo"]
            rec.update(core_size=cert["core_size"], core_edges=cert.get("core_edges", 0),
                       delta=cert["delta"], tau=cert["tau"], core_seconds=cert.get("core_seconds"),
                       max_interval_width=float(width.max()),
                       bp_seconds=round(t_bp, 2), bp_converged=bp["converged"], bp_iterations=bp["iterations"],
                       bp_violations=violations(edges, mu_bp),
                       bp_terms_outside_interval=int(np.sum(outside > 1e-12)),
                       bp_certified_error_lower_bound=float(max(outside.max(), 0.0)),
                       bp_max_abs_diff_from_core_exact=float(np.abs(mu_bp - cert["mu_hat"]).max()))
            recs.append(rec)
            print(f"{j:3d} {rec['protein']:10s} {rec['selection'][:6]} nnz={rec['n_nonzero']:4d} "
                  f"core={rec['core_size']:5d} delta={rec['delta']:.1e} core_t={rec['core_seconds']}s | "
                  f"BP outside={rec['bp_terms_outside_interval']} certErr>={rec['bp_certified_error_lower_bound']:.2e} "
                  f"maxdiff={rec['bp_max_abs_diff_from_core_exact']:.2e} conv={bp['converged']}", flush=True)

    tag = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"p1_2_certified_{args.ontology}_{key}_clip{args.clip:g}{tag}.json"
    with open(out, "w") as fh:
        json.dump(dict(args=vars(args), terms=n, proteins=recs), fh, indent=1)
    print(f"written {out}")


if __name__ == "__main__":
    main()
