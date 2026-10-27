"""P1.0a  Are realistic posteriors "hard"? BP and Gibbs against exact marginals.

Purpose (decides the paper's framing, see plan section 7): if loopy BP is near-exact on
these theories, "exact / high-accuracy WMC" needs a different justification; if BP has
real errors where paths reconverge, the accuracy argument stands. Exact marginals come
from a d4 decision-DNNF compiled once per instance and evaluated for every score vector.

Instances
  --subgraph-size 0   the full ontology (only feasible where d4 finishes: GO cc)
  --subgraph-size k   ancestor-closed sub-ontologies of about k terms (union of random terms
                      and all their ancestors), so reconvergence is preserved
Scores
  synthetic families (uniform, beta22, beta0505, sparse) and, once P1.0c has run,
  real DiamondScore vectors (--diamond <npz>), restricted to the instance's terms.

Output: results/p1_0a_<ontology>_<size>[_<tag>].json
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from wmc_common import RESULTS_DIR, load_theory, write_dimacs
from wmc_inference import (CLIP, Circuit, compile_with_d4, gibbs_marginals, loopy_bp,
                           make_scores, violations)


def ancestor_closed_subgraph(n, edges, size, rng):
    parents = [[] for _ in range(n)]
    for c, q in edges:
        parents[c].append(int(q))
    keep: set[int] = set()
    for v in rng.permutation(n):
        if len(keep) >= size:
            break
        stack = [int(v)]
        while stack:
            u = stack.pop()
            if u not in keep:
                keep.add(u)
                stack.extend(parents[u])
    nodes = np.array(sorted(keep))
    idx = -np.ones(n, dtype=np.int64)
    idx[nodes] = np.arange(len(nodes))
    m = (idx[edges[:, 0]] >= 0) & (idx[edges[:, 1]] >= 0)
    return nodes, idx[edges[m]].astype(np.int32)


def reconvergent_mask(n, edges):
    """Terms with a multi-parent ancestor-or-self, i.e. with two distinct upward paths."""
    parents = [[] for _ in range(n)]
    for c, q in edges:
        parents[c].append(int(q))
    memo = {}

    def rec(v):
        if v in memo:
            return memo[v]
        stack, order = [v], []
        while stack:  # iterative post-order
            u = stack[-1]
            todo = [w for w in parents[u] if w not in memo and w not in order]
            if todo and u not in order:
                stack.extend(todo)
                order.append(u)
                continue
            stack.pop()
            if u not in memo:
                memo[u] = len(parents[u]) >= 2 or any(memo.get(w, False) for w in parents[u])
        return memo[v]

    return np.array([rec(v) for v in range(n)])


def error_stats(mu, ref, mask):
    err = np.abs(mu - ref)
    out = dict(mae=float(err.mean()), max=float(err.max()), p99=float(np.quantile(err, 0.99)),
               frac_gt_1e2=float(np.mean(err > 1e-2)), frac_gt_1e3=float(np.mean(err > 1e-3)))
    if mask.any() and (~mask).any():
        out["mae_reconvergent"] = float(err[mask].mean())
        out["mae_tree_like"] = float(err[~mask].mean())
        out["max_reconvergent"] = float(err[mask].max())
        out["max_tree_like"] = float(err[~mask].max())
    return out


def score_vectors(args, nodes, rng):
    k = len(nodes)
    if args.diamond:
        key = args.diamond_key
        with np.load(args.diamond) as z:
            mat = csr_matrix((z[f"{key}_data"], z[f"{key}_indices"], z[f"{key}_indptr"]),
                             shape=(len(z["accessions"]), int(z["n_terms"])))
        sub = mat[:, nodes].tocsr()
        # proteins with at least 3 scored terms inside this instance; others are trivial here.
        # Half random, half the proteins with most intermediate scores (0.05..0.95): homology
        # scores are often near 0/1, and the uncertain proteins are where inference matters.
        cand = np.flatnonzero(np.diff(sub.indptr) >= 3)
        inter = np.array([np.sum((sub[r].data > 0.05) & (sub[r].data < 0.95)) for r in cand])
        k = min(args.proteins, len(cand))
        top = cand[np.argsort(-inter, kind="stable")[: k // 2]]
        rest = np.setdiff1d(cand, top)
        rows = np.concatenate([top, rng.choice(rest, min(k - len(top), len(rest)), replace=False)])
        for r in rows.astype(int):
            yield f"{key}:{r}", np.clip(sub[r].toarray().ravel().astype(float), CLIP, 1 - CLIP)
    else:
        for fam in args.families.split(","):
            for s in range(args.scores_per_family):
                yield f"{fam}:{s}", make_scores(k, fam, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--subgraph-size", type=int, default=0)
    ap.add_argument("--n-subgraphs", type=int, default=5)
    ap.add_argument("--families", default="uniform,beta22,beta0505,sparse")
    ap.add_argument("--scores-per-family", type=int, default=3)
    ap.add_argument("--diamond", default=None, help="npz from P1.0c")
    ap.add_argument("--diamond-key", default="prop")
    ap.add_argument("--proteins", type=int, default=12)
    ap.add_argument("--d4-timeout", type=float, default=1800)
    ap.add_argument("--gibbs-sweeps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    ids, edges = load_theory(args.ontology)
    n_full = len(ids)
    instances = []
    if args.subgraph_size == 0:
        instances.append((np.arange(n_full), edges))
    else:
        for _ in range(args.n_subgraphs):
            instances.append(ancestor_closed_subgraph(n_full, edges, args.subgraph_size, rng))

    records = []
    with tempfile.TemporaryDirectory() as td:
        for k, (nodes, sub_edges) in enumerate(instances):
            n = len(nodes)
            cnf, nnf = Path(td) / f"i{k}.cnf", Path(td) / f"i{k}.nnf"
            write_dimacs(cnf, n, sub_edges)
            parents_n = np.bincount(sub_edges[:, 0], minlength=n)
            rec = dict(instance=k, terms=n, edges=int(len(sub_edges)),
                       pct_multiparent=round(100 * float(np.mean(parents_n >= 2)), 2))
            try:
                rec["d4"] = compile_with_d4(cnf, nnf, timeout_s=args.d4_timeout)
            except Exception as exc:  # timeout or failure is a result, not a crash
                rec["d4"] = dict(failed=repr(exc)[:300])
                records.append(rec)
                print(json.dumps(rec), flush=True)
                continue
            t0 = time.time()
            circ = Circuit(nnf, n)
            rec["circuit"] = dict(nodes=circ.n_nodes, edges=circ.n_edges,
                                  parse_seconds=round(time.time() - t0, 2))
            mask = reconvergent_mask(n, sub_edges)
            rec["pct_reconvergent"] = round(100 * float(mask.mean()), 2)
            rec["runs"] = []
            for label, p in score_vectors(args, nodes, rng):
                t0 = time.time(); ref, logz = circ.marginals(p); t_ex = time.time() - t0
                t0 = time.time(); bp = loopy_bp(n, sub_edges, p); t_bp = time.time() - t0
                t0 = time.time()
                g = gibbs_marginals(n, sub_edges, p, rng, sweeps=args.gibbs_sweeps)
                t_g = time.time() - t0
                run = dict(scores=label, logZ=logz, exact_seconds=round(t_ex, 3),
                           n_intermediate=int(np.sum((p > 0.05) & (p < 0.95))),
                           bp=dict(**error_stats(bp["mu"], ref, mask), iterations=bp["iterations"],
                                   converged=bp["converged"], violations=violations(sub_edges, bp["mu"]),
                                   seconds=round(t_bp, 2)),
                           gibbs=dict(**error_stats(g["mu"], ref, mask),
                                      max_bracket_gap=float(g["bracket_gap"].max()),
                                      max_chain_se=float(g["chain_se"].max()), seconds=round(t_g, 2)))
                rec["runs"].append(run)
                print(f"inst {k} n={n} {label:12s} BP mae={run['bp']['mae']:.2e} max={run['bp']['max']:.2e} "
                      f"conv={bp['converged']} | Gibbs mae={run['gibbs']['mae']:.2e} max={run['gibbs']['max']:.2e} "
                      f"| exact {t_ex:.2f}s", flush=True)
            records.append(rec)

    tag = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"p1_0a_{args.ontology}_{args.subgraph_size}{tag}.json"
    with open(out, "w") as fh:
        json.dump(dict(args=vars(args), instances=records), fh, indent=1)
    print(f"written {out}")


if __name__ == "__main__":
    main()
