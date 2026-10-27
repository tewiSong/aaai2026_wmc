"""P0.4  Determinism profile: how many bag assignments are logically possible?

Purpose: a dense junction tree pays 2^|S| per bag S, but under is-a implications only
assignments that are up-sets of the reachability order restricted to S can have nonzero
mass (every such up-set extends to a valid full assignment). If these counts are far
below 2^|S|, a determinism-aware exact method (sparse tables, d-DNNF compilation) may
scale far beyond what the manuscript's dense-JT OOM rows suggest - a threat to the
scale claim, and also a possible route to exact ground truth for P2.

For each bag with >= --bag-min vertices from P0.3 we report
  |S|, Dilworth width of the induced order, log2 of the chain-cover bound
  prod_i (|C_i| + 1) >= #up-sets, and the exact log2 #up-sets when it can be counted.

Usage
  p0_4_determinism_profile_of_treewidth_bags.py --ontology go_mf [--heuristic min-fill]
Output
  results/p0_4_determinism_<ontology>_<heuristic>.json (summary + per-bag rows)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from wmc_common import ORDER_DIR, RESULTS_DIR, load_theory

sys.setrecursionlimit(100000)


class _Budget(Exception):
    pass


def count_upsets(comp: list[int], budget: int) -> int:
    """Number of antichains (= up-sets) of a poset given comparability bitmasks."""
    memo: dict[int, int] = {}
    calls = [0]

    def rec(mask: int) -> int:
        if mask == 0:
            return 1
        hit = memo.get(mask)
        if hit is not None:
            return hit
        calls[0] += 1
        if calls[0] > budget:
            raise _Budget
        x = (mask & -mask).bit_length() - 1
        rest = mask & ~(1 << x)
        val = rec(rest) + rec(rest & ~comp[x])
        memo[mask] = val
        return val

    return rec((1 << len(comp)) - 1)


def min_chain_cover(less: list[int], k: int) -> list[int]:
    """Dilworth: minimum chain partition via maximum bipartite matching (Hopcroft-Karp)
    on the strict order x < y of a transitively closed relation. Returns chain lengths."""
    rows, cols = [], []
    for i in range(k):
        m = less[i]
        while m:
            j = (m & -m).bit_length() - 1
            rows.append(i)
            cols.append(j)
            m &= m - 1
    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(k, k))
    nxt = maximum_bipartite_matching(graph, perm_type="column").tolist()  # row i -> col
    has_pred = set(j for j in nxt if j != -1)
    lengths = []
    for i in range(k):
        if i in has_pred:
            continue
        length, cur = 0, i
        while cur != -1:
            length += 1
            cur = nxt[cur]
        lengths.append(length)
    return lengths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--heuristic", default="min-fill")
    ap.add_argument("--max-bags", type=int, default=5000,
                    help="analyse the largest bags first, up to this many")
    ap.add_argument("--count-budget", type=int, default=200_000,
                    help="recursion budget for the exact up-set count per bag")
    args = ap.parse_args()

    ids, edges = load_theory(args.ontology)
    n = len(ids)
    parents = [[] for _ in range(n)]
    for c, p in edges:
        parents[c].append(int(p))

    tag = f"{args.ontology}_{args.heuristic}"
    with np.load(ORDER_DIR / f"{tag}.npz") as z:
        sizes, flat = z["bag_sizes"], z["bag_flat"]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    bag_ix = np.argsort(-sizes, kind="stable")[: args.max_bags]

    anc_cache: dict[int, frozenset] = {}

    def ancestors(v: int) -> frozenset:
        hit = anc_cache.get(v)
        if hit is None:
            seen, dq = set(), deque(parents[v])
            while dq:
                u = dq.popleft()
                if u not in seen:
                    seen.add(u)
                    dq.extend(parents[u])
            hit = anc_cache[v] = frozenset(seen)
        return hit

    rows, t0 = [], time.time()
    for b in bag_ix:
        S = flat[offsets[b]:offsets[b + 1]].tolist()
        k = len(S)
        pos = {v: i for i, v in enumerate(S)}
        less = [0] * k  # less[i] bit j: S[i] => S[j] (j strict ancestor of i)
        for i, v in enumerate(S):
            for a in ancestors(v):
                j = pos.get(a)
                if j is not None:
                    less[i] |= 1 << j
        greater = [0] * k
        for i in range(k):
            m = less[i]
            while m:
                j = (m & -m).bit_length() - 1
                greater[j] |= 1 << i
                m &= m - 1
        comp = [less[i] | greater[i] for i in range(k)]
        chains = min_chain_cover(less, k)
        log2_bound = sum(math.log2(c + 1) for c in chains)
        try:
            exact = count_upsets(comp, args.count_budget)
            log2_exact = math.log2(exact)
        except _Budget:
            log2_exact = None
        rows.append(dict(size=k, dilworth_width=len(chains), log2_chain_bound=round(log2_bound, 3),
                         log2_exact_upsets=None if log2_exact is None else round(log2_exact, 3)))

    def best(r):
        return r["log2_exact_upsets"] if r["log2_exact_upsets"] is not None else r["log2_chain_bound"]

    summary = dict(
        ontology=args.ontology, heuristic=args.heuristic, bags_analysed=len(rows),
        bags_total=int(len(sizes)), seconds=round(time.time() - t0, 1),
        max_bag_size=max((r["size"] for r in rows), default=0),
        max_log2_states=max((best(r) for r in rows), default=0.0),
        # an antichain of size w alone gives 2^w up-sets, so width is a rigorous lower bound
        max_log2_states_lower_bound=max((r["dilworth_width"] for r in rows), default=0),
        max_dilworth_width=max((r["dilworth_width"] for r in rows), default=0),
        exact_counted_fraction=round(np.mean([r["log2_exact_upsets"] is not None for r in rows]), 3)
        if rows else None,
        note="dense JT cost ~ 2^max_bag_size; determinism-aware cost ~ 2^max_log2_states "
             "(log2_chain_bound is a rigorous upper bound on log2 #up-sets)",
    )
    out = RESULTS_DIR / f"p0_4_determinism_{tag}.json"
    with open(out, "w") as fh:
        json.dump(dict(summary=summary, bags=rows), fh, indent=1)
    print(json.dumps(summary, indent=2))
    print("largest bags (size, width, log2 bound, log2 exact):")
    for r in rows[:15]:
        print(f"  {r['size']:5d} {r['dilworth_width']:5d} {r['log2_chain_bound']:9.2f} {r['log2_exact_upsets']}")
    print(f"written {out}")


if __name__ == "__main__":
    main()
