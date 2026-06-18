"""Exp 3b: top-k provenance (Scallop) drifts at reconvergences.

Scallop keeps only the k best proofs per fact. For the true-path upward rule
  holds(P) :- sub(C,P), holds(C).
a proof that a term t holds is any single asserted descendant fact x(s) with s = t or a
descendant of t (one descendant suffices), so the exact provenance marginal is
  P(holds(t)) = 1 - prod_{s in {t} U descendants(t)} (1 - p_s),
and Scallop's top-k-proofs keeps only the k descendants of largest probability:
  P_k(holds(t)) = 1 - prod over the k largest p_s of (1 - p_s).

The undercount therefore grows with proof multiplicity (the number of descendants =
reconvergence) and shrinks as k admits more proofs - the same reconvergence axis along
which the soft closure departs from exact WMC. This reproduces the four-node reconvergence
(exact 0.9375, 0.50 at k=1, 0.9375 at k=10) and measures drift on real GO neighborhoods.

This implements Scallop's top-k-proofs provenance semantics directly; it needs no GPU.
"""

from __future__ import annotations

import json
import os

import numpy as np

import lib


def descendants(theory):
    """Map each term to the set of its descendants (terms that imply it), inclusive."""
    children = {}  # parent -> [children that imply it]
    for c, p in theory.nf1:
        children.setdefault(p, []).append(c)
    desc = {}

    import sys
    sys.setrecursionlimit(1000000)
    from functools import lru_cache

    # iterative DFS to avoid recursion limits on deep DAGs
    def compute(root):
        stack = [root]
        seen = set()
        while stack:
            u = stack.pop()
            for c in children.get(u, []):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return seen

    return compute, children


def exact_and_topk(p_list, ks):
    """Exact provenance OR-marginal and top-k versions for a list of fact probabilities."""
    arr = np.sort(np.array(p_list))[::-1]
    exact = 1.0 - np.prod(1.0 - arr)
    out = {}
    for k in ks:
        out[k] = float(1.0 - np.prod(1.0 - arr[:k])) if k <= len(arr) else float(exact)
    return float(exact), out


def four_node_demo():
    """The canonical four-node reconvergence at edge-probability 1/2."""
    # holds(a) proofs: x(a),x(b),x(c),x(d) each 1/2
    ks = [1, 2, 3, 5, 10]
    exact, topk = exact_and_topk([0.5, 0.5, 0.5, 0.5], ks)
    print(f"[scallop] four-node reconvergence exact={exact:.4f}", flush=True)
    for k in ks:
        print(f"    k={k:2d}: {topk[k]:.4f}  (drift {exact - topk[k]:+.4f})", flush=True)
    return dict(exact=exact, topk={str(k): topk[k] for k in ks})


def go_neighborhoods(seed=0, n_targets=200, ks=(1, 2, 3, 5, 10, 20, 50)):
    """Drift on real bp neighborhoods: pick high-reconvergence targets and measure max
    drift vs k against the exact provenance marginal."""
    theories = lib.load_theories()
    th = theories["bp"]
    priors = lib.make_priors(th, seed=seed)
    compute_desc, _children = descendants(th)

    # rank targets by number of descendants (proof multiplicity)
    # to keep it tractable, sample terms and compute descendant counts
    rng = np.random.default_rng(seed)
    sample = list(th.atoms)
    rng.shuffle(sample)

    targets = []
    for t in sample:
        d = compute_desc(t)
        if len(d) >= 5:
            targets.append((t, d))
        if len(targets) >= n_targets:
            break

    drift_by_k = {k: [] for k in ks}
    rows = []
    for t, d in targets:
        facts = [priors[s] for s in ([t] + list(d))]
        exact, topk = exact_and_topk(facts, ks)
        for k in ks:
            drift_by_k[k].append(abs(exact - topk[k]))
        rows.append(dict(term=t, n_proofs=len(facts), exact=exact,
                         topk={str(k): topk[k] for k in ks}))

    summary = {str(k): dict(mean_drift=float(np.mean(drift_by_k[k])),
                            max_drift=float(np.max(drift_by_k[k]))) for k in ks}
    print(f"[scallop] bp neighborhoods ({len(targets)} targets):", flush=True)
    for k in ks:
        print(f"    k={k:2d}: mean_drift={summary[str(k)]['mean_drift']:.4f} "
              f"max_drift={summary[str(k)]['max_drift']:.4f}", flush=True)
    return dict(n_targets=len(targets), summary=summary)


def main():
    lib.ensure_results_dir()
    out = dict(four_node=four_node_demo(), bp_neighborhoods=go_neighborhoods())
    path = os.path.join(lib.RESULTS_DIR, "exp3b_scallop_topk.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
