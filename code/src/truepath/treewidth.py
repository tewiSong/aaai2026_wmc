"""Min-fill treewidth heuristic and elimination orderings.

Treewidth governs the cost Theta(m * 2^tw) of exact WMC by knowledge compilation.
We compute the standard min-fill upper bound: repeatedly eliminate the vertex whose
elimination adds the fewest fill edges, making its neighbourhood a clique, and report
the largest clique (= largest induced bag) minus one.

The implementation maintains per-vertex fill counts incrementally and uses a lazy
priority queue: eliminating a vertex only changes the fill counts of vertices within
distance two of it, so only those are recomputed and re-pushed. Stale heap entries are
skipped via a version stamp. This makes min-fill feasible on the 21k-node bp graph.

The min-fill order is also returned so the same ordering can drive the SDD vtree and
the junction-tree construction (exact WMC), keeping the reported tw consistent with
the circuit actually built.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Set, Tuple

import networkx as nx


def _fill_count(adj: Dict[object, Set[object]], v: object) -> int:
    """Number of non-adjacent pairs among v's current neighbours (fill edges added)."""
    nbrs = list(adj[v])
    missing = 0
    for i in range(len(nbrs)):
        ai = adj[nbrs[i]]
        for j in range(i + 1, len(nbrs)):
            if nbrs[j] not in ai:
                missing += 1
    return missing


def min_fill_order(graph: nx.Graph) -> Tuple[List, int]:
    """Return (elimination_order, width) under the implemented min-fill heuristic.

    width is the treewidth upper bound: max over eliminated vertices of the size of its live
    neighbourhood at elimination time (= largest bag size - 1).

    The adjacency is a dense NumPy boolean matrix, so a fill count is
        fill(v) = C(d,2) - (edges among N(v)),  edges among N(v) = A[N(v),N(v)].sum()/2,
    a single vectorised submatrix sum, and cliquing N(v) is one boolean broadcast. After
    eliminating v the only fill counts that change are those of vertices within distance two,
    which are recomputed exactly. This is the standard min-fill heuristic (it matches NetworkX's
    `treewidth_min_fill_in` up to heuristic tie-breaking) with no lazy / min-degree
    substitute; the vectorisation makes the fill-count computation fast on the dense bp graph.
    """
    import numpy as np

    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return [], 0
    idx = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((n, n), dtype=bool)
    for u, w in graph.edges():
        i, j = idx[u], idx[w]
        if i != j:
            A[i, j] = True
            A[j, i] = True

    def fill_of(v: int) -> int:
        nb = np.flatnonzero(A[v])
        d = nb.size
        if d < 2:
            return 0
        e = int(A[np.ix_(nb, nb)].sum()) // 2
        return d * (d - 1) // 2 - e

    fill = [fill_of(i) for i in range(n)]
    version = [0] * n
    alive = np.ones(n, dtype=bool)
    heap: List[Tuple[int, int, int, int]] = [(fill[i], int(A[i].sum()), 0, i) for i in range(n)]
    heapq.heapify(heap)

    order: List = []
    width = 0
    remaining = n
    while remaining:
        while True:
            f, deg, ver, v = heapq.heappop(heap)
            if alive[v] and ver == version[v]:
                break
        nb = np.flatnonzero(A[v])
        width = max(width, nb.size)
        # Vertices whose fill can change: the 2-hop neighbourhood of v (a correct superset).
        if nb.size:
            twohop = np.flatnonzero(A[nb].any(axis=0))
        else:
            twohop = nb
        # Make N(v) a clique, then detach v.
        if nb.size:
            A[np.ix_(nb, nb)] = True
            A[nb, nb] = False  # keep diagonal clear
        A[v, :] = False
        A[:, v] = False
        alive[v] = False
        for u in twohop:
            u = int(u)
            if u != v and alive[u]:
                fill[u] = fill_of(u)
                version[u] += 1
                heapq.heappush(heap, (fill[u], int(A[u].sum()), version[u], u))
        order.append(nodes[v])
        remaining -= 1
    return order, width


def min_fill_width_bounded(graph: nx.Graph, limit: int) -> int:
    """Min-fill width, stopping once the width is known to exceed `limit`.

    This follows the same elimination rule as `min_fill_order`. If the returned value is
    at most `limit`, it is the complete min-fill width. If it is larger than `limit`, the
    exact larger width is intentionally not computed because the caller only needs to
    reject that graph under the cap.
    """
    import numpy as np

    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return 0
    idx = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((n, n), dtype=bool)
    for u, w in graph.edges():
        i, j = idx[u], idx[w]
        if i != j:
            A[i, j] = True
            A[j, i] = True

    def fill_of(v: int) -> int:
        nb = np.flatnonzero(A[v])
        d = nb.size
        if d < 2:
            return 0
        e = int(A[np.ix_(nb, nb)].sum()) // 2
        return d * (d - 1) // 2 - e

    fill = [fill_of(i) for i in range(n)]
    version = [0] * n
    alive = np.ones(n, dtype=bool)
    heap: List[Tuple[int, int, int, int]] = [(fill[i], int(A[i].sum()), 0, i) for i in range(n)]
    heapq.heapify(heap)

    width = 0
    remaining = n
    while remaining:
        while True:
            f, deg, ver, v = heapq.heappop(heap)
            if alive[v] and ver == version[v]:
                break
        nb = np.flatnonzero(A[v])
        width = max(width, nb.size)
        if width > limit:
            return int(width)
        if nb.size:
            twohop = np.flatnonzero(A[nb].any(axis=0))
            A[np.ix_(nb, nb)] = True
            A[nb, nb] = False
        else:
            twohop = nb
        A[v, :] = False
        A[:, v] = False
        alive[v] = False
        for u in twohop:
            u = int(u)
            if u != v and alive[u]:
                fill[u] = fill_of(u)
                version[u] += 1
                heapq.heappush(heap, (fill[u], int(A[u].sum()), version[u], u))
        remaining -= 1
    return int(width)


def treewidth_min_fill(graph: nx.Graph) -> int:
    """Min-fill treewidth upper bound (max over connected components).

    No fallback: every component uses the same min-fill ordering rule above; large dense
    components (bp) are given the wall-clock they need.
    """
    tw = 0
    for comp in nx.connected_components(graph):
        sub = graph.subgraph(comp)
        if sub.number_of_nodes() <= 1:
            continue
        _, w = min_fill_order(sub)
        tw = max(tw, w)
    return tw


def component_treewidths(graph: nx.Graph) -> List[Tuple[int, int]]:
    """Return (n_nodes, treewidth) per connected component, sorted by tw descending."""
    out: List[Tuple[int, int]] = []
    for comp in nx.connected_components(graph):
        sub = graph.subgraph(comp)
        if sub.number_of_nodes() <= 1:
            out.append((sub.number_of_nodes(), 0))
            continue
        _, w = min_fill_order(sub)
        out.append((sub.number_of_nodes(), w))
    out.sort(key=lambda t: t[1], reverse=True)
    return out
