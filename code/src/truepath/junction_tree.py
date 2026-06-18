"""Exact WMC marginals by junction-tree (Shafer-Shenoy) calibration.

Given the true-path clause set and a min-fill elimination order, we build a junction
tree of the moralized constraint graph, place the clause potentials and unary priors on
cliques, and calibrate with two-pass sum-product message passing. Reading each clique's
calibrated potential yields exact per-variable marginals mu_i = P(y_i = 1 | T) in a
single calibration, at cost O(n * 2^treewidth).

This is the exact reference used for the cc namespace (whole graph, tw 9) and for every
low-treewidth module of the modular construction. It is exact arithmetic throughout and
is validated against brute force and the SDD path.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .treewidth import min_fill_order


def _build_cliques(graph, order: Sequence) -> List[Tuple]:
    """Simulate elimination along `order`, returning a clique (as a sorted tuple of
    nodes) for each eliminated vertex."""
    adj: Dict[object, set] = {v: set(graph.neighbors(v)) for v in graph.nodes()}
    cliques: List[Tuple] = []
    pos = {v: i for i, v in enumerate(order)}
    for v in order:
        nbrs = [u for u in adj[v] if u in adj]
        clique = tuple(sorted([v] + nbrs, key=lambda x: pos[x]))
        cliques.append(clique)
        # add fill edges
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                adj[nbrs[i]].add(nbrs[j])
                adj[nbrs[j]].add(nbrs[i])
        for u in nbrs:
            adj[u].discard(v)
        del adj[v]
    return cliques


class JunctionTree:
    """A calibrated junction tree over a connected constraint (sub)graph."""

    def __init__(self, graph, atoms_global_index: Dict[object, int]):
        # atoms_global_index maps node -> global variable index (1-based) for clauses.
        self.graph = graph
        self.gidx = atoms_global_index
        order, self.width = min_fill_order(graph)
        self.order = order
        self.pos = {v: i for i, v in enumerate(order)}
        raw = _build_cliques(graph, order)
        # Keep only maximal cliques but retain mapping for parent assignment.
        self.clique_of_elim = raw  # clique created when order[i] eliminated
        # Build clique nodes (use index i as id). Parent: earliest-eliminated other member.
        self.cliques: List[frozenset] = [frozenset(c) for c in raw]
        self.parent: List[int] = [-1] * len(raw)
        for i, c in enumerate(raw):
            v = order[i]
            others = [u for u in c if u != v]
            if not others:
                self.parent[i] = -1
                continue
            # earliest-eliminated other member w; its clique is clique_of_elim at pos[w]
            w = min(others, key=lambda u: self.pos[u])
            self.parent[i] = self.pos[w]
        self.children: List[List[int]] = [[] for _ in range(len(raw))]
        for i, p in enumerate(self.parent):
            if p >= 0:
                self.children[p].append(i)
        self.roots = [i for i, p in enumerate(self.parent) if p < 0]

    def calibrate_marginals(
        self,
        clauses: Sequence[Tuple[int, ...]],
        priors: Dict[object, float],
    ) -> Dict[object, float]:
        """Return exact marginals P(node=1) for every node in this graph.

        `clauses` are global signed-index clauses; only those whose variables all lie in
        this graph are placed here. `priors[node]` is the prior of each node.
        """
        nodes = list(self.graph.nodes())
        node_of_gidx = {self.gidx[v]: v for v in nodes}
        local_set = set(self.gidx[v] for v in nodes)

        # Clique potentials as numpy tables; var order within clique = sorted by pos.
        clique_vars: List[List[object]] = [sorted(self.cliques[i], key=lambda x: self.pos[x])
                                            for i in range(len(self.cliques))]
        pot: List[np.ndarray] = [np.ones((2,) * len(cv), dtype=np.float64) for cv in clique_vars]
        var_axis: List[Dict[object, int]] = [{v: k for k, v in enumerate(cv)} for cv in clique_vars]

        def find_clique(scope: List[object]) -> int:
            sset = set(scope)
            for i, c in enumerate(self.cliques):
                if sset <= c:
                    return i
            raise RuntimeError(f"no clique contains scope {scope}")

        # Place unary priors.
        for v in nodes:
            ci = find_clique([v])
            ax = var_axis[ci][v]
            p = priors[v]
            shape = [1] * pot[ci].ndim
            shape[ax] = 2
            factor = np.array([1.0 - p, p]).reshape(shape)
            pot[ci] = pot[ci] * factor

        # Place clause potentials.
        for clause in clauses:
            gvars = [abs(l) for l in clause]
            if any(gv not in local_set for gv in gvars):
                continue
            scope = [node_of_gidx[gv] for gv in gvars]
            ci = find_clique(scope)
            cv = clique_vars[ci]
            want = {abs(l): (1 if l > 0 else 0) for l in clause}
            tab = np.ones((2,) * len(cv), dtype=np.float64)
            # set to 0 the single local pattern violating the clause: all literals false
            # general: factor over clause vars = indicator(clause satisfied), broadcast.
            sub_axes = [var_axis[ci][node_of_gidx[gv]] for gv in gvars]
            # build small indicator over clause vars then broadcast-multiply
            small = np.zeros((2,) * len(gvars), dtype=np.float64)
            for idx in np.ndindex(*small.shape):
                assign = {gvars[k]: idx[k] for k in range(len(gvars))}
                sat = any(assign[gv] == want[gv] for gv in gvars)
                small[idx] = 1.0 if sat else 0.0
            # expand small into clique shape
            shape = [1] * pot[ci].ndim
            for k, gv in enumerate(gvars):
                shape[sub_axes[k]] = 2
            # need to place small's axes onto sub_axes in order
            perm = np.moveaxis(small, range(len(gvars)),
                               sorted(range(len(gvars)), key=lambda k: sub_axes[k]))
            # Build by iterating: simplest robust approach -> use einsum-like broadcast
            big = np.ones((2,) * len(cv), dtype=np.float64)
            it_axes = sorted(range(len(gvars)), key=lambda k: sub_axes[k])
            # reorder small axes to ascending sub_axis
            small_sorted = np.transpose(small, it_axes)
            bshape = [1] * len(cv)
            for k in it_axes:
                bshape[sub_axes[k]] = 2
            big = small_sorted.reshape(bshape)
            pot[ci] = pot[ci] * big

        # Separators between clique i and parent p: intersection vars.
        # Two-pass calibration (collect upward, distribute downward) - Shafer-Shenoy
        # implemented on potentials with Hugin-style updates.
        msg_up: Dict[Tuple[int, int], np.ndarray] = {}

        def marginalize(table: np.ndarray, cv: List[object], keep: List[object]) -> np.ndarray:
            keepset = set(keep)
            axes = tuple(k for k, v in enumerate(cv) if v not in keepset)
            out = table.sum(axis=axes) if axes else table
            return out

        order_post: List[int] = []
        visited = [False] * len(self.cliques)

        def post(i):
            visited[i] = True
            for c in self.children[i]:
                post(c)
            order_post.append(i)
        import sys
        sys.setrecursionlimit(1000000)
        for r in self.roots:
            post(r)

        # Collect: leaves -> root
        msg: Dict[Tuple[int, int], Tuple[List[object], np.ndarray]] = {}
        for i in order_post:
            p = self.parent[i]
            if p < 0:
                continue
            sep = sorted(self.cliques[i] & self.cliques[p], key=lambda x: self.pos[x])
            # combine potential with incoming child messages already folded into pot[i]
            m = marginalize(pot[i], clique_vars[i], sep)
            msg[(i, p)] = (sep, m)
            # fold into parent potential
            pot[p] = _multiply_in(pot[p], clique_vars[p], var_axis[p], sep, m)

        # Distribute: root -> leaves
        for i in reversed(order_post):
            for c in self.children[i]:
                sep = sorted(self.cliques[i] & self.cliques[c], key=lambda x: self.pos[x])
                # message to child = marginalize parent's potential / message it sent up
                up_sep, up_m = msg[(c, i)]
                par_marg = marginalize(pot[i], clique_vars[i], sep)
                down = par_marg / np.where(up_m > 0, up_m, 1.0)
                pot[c] = _multiply_in(pot[c], clique_vars[c], var_axis[c], sep, down)

        # Read marginals.
        marg: Dict[object, float] = {}
        for v in nodes:
            ci = find_clique([v])
            table = pot[ci]
            cv = clique_vars[ci]
            m = marginalize(table, cv, [v])
            s = m.sum()
            ax = 1  # after marginalizing to single var, index 1 = true
            marg[v] = float(m[1] / s) if s > 0 else 0.0
        return marg


def _multiply_in(table: np.ndarray, cv: List[object], var_axis: Dict[object, int],
                 sep: List[object], sep_table: np.ndarray) -> np.ndarray:
    """Multiply `table` (over clique vars cv) by `sep_table` (over sep vars, sorted by pos)."""
    if len(sep) == 0:
        return table * sep_table  # scalar
    bshape = [1] * table.ndim
    # sep_table axes are in the order given by `sep`; map to clique axes
    sep_axes = [var_axis[v] for v in sep]
    order = sorted(range(len(sep)), key=lambda k: sep_axes[k])
    st = np.transpose(sep_table, order)
    for k in order:
        bshape[sep_axes[k]] = 2
    return table * st.reshape(bshape)
