"""P0.3  Whole-theory induced width (min-fill and min-degree) vs the manuscript.

Purpose: the manuscript's scale argument rests on whole-theory min-fill widths
12 / 37 / 269 / 46 / 738 (GO cc / mf / bp / UBERON / ChEBI). This recomputes them on the
P0.2 theories and stores the elimination order and the large bags (cliques), which P0.4
uses for the determinism profile and P1 uses to configure tree-decomposition baselines.

Usage
  p0_3_minfill_width_check_vs_paper.py --ontology go_cc [--heuristic min-fill|min-degree]
  p0_3_minfill_width_check_vs_paper.py --self-test
Outputs
  data/orders/<ontology>_<heuristic>.npz  (order, bags with >= --bag-min vertices)
  results/p0_3_width_<ontology>_<heuristic>.json
"""
from __future__ import annotations

import argparse
import heapq
import json
import random
import time

import numpy as np

from wmc_common import ORDER_DIR, PAPER_STRUCTURE, RESULTS_DIR, load_theory, undirected_adjacency


def initial_fill(adj: list[set[int]], v: int) -> int:
    nb = adj[v]
    d = len(nb)
    inner = sum(len(adj[u] & nb) for u in nb) // 2
    return d * (d - 1) // 2 - inner


def eliminate_greedy(adj: list[set[int]], heuristic: str, bag_min: int = 16,
                     time_limit: float = float("inf"), check: bool = False) -> dict:
    """Greedy elimination with exact incremental fill-in counts. Mutates adj."""
    n = len(adj)
    alive = np.ones(n, dtype=bool)
    use_fill = heuristic == "min-fill"
    fill = [initial_fill(adj, v) for v in range(n)] if use_fill else None
    stamp = [0] * n

    def key(v):
        return (fill[v], len(adj[v]), v) if use_fill else (len(adj[v]), v)

    heap = [(key(v), v, 0) for v in range(n)]
    heapq.heapify(heap)
    order, bags, width, remaining = [], [], -1, n
    t0, complete = time.time(), True

    while remaining:
        # once remaining-1 <= width no later bag can exceed the width found so far
        if remaining - 1 <= width:
            rest = np.flatnonzero(alive).tolist()
            order.extend(rest)
            if len(rest) >= bag_min:
                bags.append(rest)
            break
        if time.time() - t0 > time_limit:
            complete = False
            break
        _, v, s = heapq.heappop(heap)
        if not alive[v] or s != stamp[v]:
            continue

        nb = adj[v]
        width = max(width, len(nb))
        if len(nb) + 1 >= bag_min:
            bags.append([v, *nb])
        order.append(v)
        alive[v] = False
        remaining -= 1
        touched = set(nb)
        nbl = list(nb)
        v_has_fill = (fill[v] > 0) if use_fill else True

        # 1) remove v; pairs (v, x) with x not adjacent to v leave fill(u)
        for u in nbl:
            adj[u].discard(v)
            if use_fill:
                fill[u] -= len(adj[u]) - len(adj[u] & nb)

        # 2) turn N(v) into a clique, updating fill counts edge by edge
        if v_has_fill:
            for i, a in enumerate(nbl):
                adj_a = adj[a]
                for b in nbl[i + 1:]:
                    if b in adj_a:
                        continue
                    if use_fill:
                        common = adj_a & adj[b]
                        for w in common:
                            fill[w] -= 1
                        touched.update(common)
                        fill[a] += len(adj_a) - len(common)
                        fill[b] += len(adj[b]) - len(common)
                    adj_a.add(b)
                    adj[b].add(a)
        adj[v] = set()

        for u in touched:
            if alive[u]:
                stamp[u] += 1
                heapq.heappush(heap, (key(u), u, stamp[u]))
        if check and use_fill:
            for u in np.flatnonzero(alive):
                assert fill[u] == initial_fill(adj, u), (u, fill[u], initial_fill(adj, u))

    return dict(width=width, order=order, bags=bags, complete=complete,
                seconds=round(time.time() - t0, 2))


def self_test() -> None:
    """Incremental fill counts must equal recomputed counts on random graphs, and both
    heuristics must give exact widths on trees (1), cycles (2) and cliques (n-1)."""
    rng = random.Random(0)
    for trial in range(200):
        n = rng.randint(2, 30)
        edges = {(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(0, 3 * n))}
        edges = np.array([(a, b) for a, b in edges if a != b] or [(0, 1)], dtype=np.int32)
        for h in ("min-fill", "min-degree"):
            res = eliminate_greedy(undirected_adjacency(n, edges), h, check=True)
            assert sorted(res["order"]) == list(range(n)), trial
    for n in (5, 12):
        path = np.array([(i, i + 1) for i in range(n - 1)], dtype=np.int32)
        cyc = np.vstack([path, [[n - 1, 0]]]).astype(np.int32)
        clique = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        for h in ("min-fill", "min-degree"):
            assert eliminate_greedy(undirected_adjacency(n, path), h)["width"] == 1
            assert eliminate_greedy(undirected_adjacency(n, cyc), h)["width"] == 2
            assert eliminate_greedy(undirected_adjacency(n, clique), h)["width"] == n - 1
    print("self-test passed: incremental fill counts and widths verified")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology")
    ap.add_argument("--heuristic", default="min-fill", choices=["min-fill", "min-degree"])
    ap.add_argument("--bag-min", type=int, default=16)
    ap.add_argument("--time-limit-hours", type=float, default=float("inf"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return

    ids, edges = load_theory(args.ontology)
    adj = undirected_adjacency(len(ids), edges)
    res = eliminate_greedy(adj, args.heuristic, args.bag_min, args.time_limit_hours * 3600)

    ORDER_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.ontology}_{args.heuristic}"
    sizes = np.array([len(b) for b in res["bags"]], dtype=np.int32)
    np.savez_compressed(ORDER_DIR / f"{tag}.npz", order=np.array(res["order"], dtype=np.int32),
                        bag_sizes=sizes,
                        bag_flat=np.concatenate([np.array(b, dtype=np.int32) for b in res["bags"]])
                        if res["bags"] else np.zeros(0, np.int32))
    paper = PAPER_STRUCTURE.get(args.ontology, {}).get("minfill_width")
    summary = dict(ontology=args.ontology, heuristic=args.heuristic, terms=len(ids),
                   width=res["width"], complete=res["complete"], seconds=res["seconds"],
                   paper_minfill_width=paper, n_bags_ge_bag_min=len(sizes),
                   bag_size_quantiles={q: int(np.quantile(sizes, q)) for q in (0.5, 0.9, 0.99, 1.0)}
                   if len(sizes) else {})
    with open(RESULTS_DIR / f"p0_3_width_{tag}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    if not res["complete"]:
        print("INCOMPLETE: time limit reached; the width above is a partial maximum, not a bound")


if __name__ == "__main__":
    main()
