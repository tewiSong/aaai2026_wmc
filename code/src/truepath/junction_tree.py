"""Exact WMC marginals by junction-tree (Shafer-Shenoy) calibration.

Given the true-path clause set and a min-fill elimination order, we build a junction
tree of the moralized constraint graph, place the clause potentials and unary priors on
cliques, and calibrate with two-pass sum-product message passing. Reading each clique's
calibrated potential yields exact per-variable marginals mu_i = P(y_i = 1 | T) in a
single calibration, at cost O(n * 2^treewidth).

This is the exact reference used for the cc namespace (whole graph, tw 12) and for every
low-treewidth module of the modular construction. The factorization is exact and evaluated
in floating point; outputs are validated against brute force and the SDD path.
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

        neighbors: List[List[int]] = [[] for _ in range(len(self.cliques))]
        for i, p in enumerate(self.parent):
            if p >= 0:
                neighbors[i].append(p)
                neighbors[p].append(i)

        def separator(i: int, j: int) -> List[object]:
            return sorted(self.cliques[i] & self.cliques[j], key=lambda x: self.pos[x])

        # Shafer-Shenoy messages: m_{i->j} = sum_{C_i \ S_ij} phi_i * prod_{k != j} m_{k->i}.
        messages: Dict[Tuple[int, int], Tuple[List[object], np.ndarray]] = {}

        def compute_message(i: int, j: int) -> None:
            table = pot[i].copy()
            for k in neighbors[i]:
                if k == j:
                    continue
                if (k, i) not in messages:
                    raise RuntimeError(f"missing message {k}->{i} while computing {i}->{j}")
                sep_ki, msg_ki = messages[(k, i)]
                table = _multiply_in(table, clique_vars[i], var_axis[i], sep_ki, msg_ki)
            sep_ij = separator(i, j)
            msg = marginalize(table, clique_vars[i], sep_ij)
            if not np.all(np.isfinite(msg)):
                raise FloatingPointError(f"non-finite JT message {i}->{j}")
            z = float(msg.sum())
            if not np.isfinite(z) or z <= 0.0:
                raise FloatingPointError(f"zero JT message mass {i}->{j} over separator {sep_ij}")
            msg = msg / z
            messages[(i, j)] = (sep_ij, msg)

        # Collect: leaves -> root.
        for i in order_post:
            p = self.parent[i]
            if p < 0:
                continue
            compute_message(i, p)

        # Distribute: root -> leaves.
        for i in reversed(order_post):
            for c in self.children[i]:
                compute_message(i, c)

        # Read marginals.
        marg: Dict[object, float] = {}
        for v in nodes:
            ci = find_clique([v])
            table = pot[ci].copy()
            cv = clique_vars[ci]
            for k in neighbors[ci]:
                sep_ki, msg_ki = messages[(k, ci)]
                table = _multiply_in(table, cv, var_axis[ci], sep_ki, msg_ki)
            m = marginalize(table, cv, [v])
            s = m.sum()
            ax = 1  # after marginalizing to single var, index 1 = true
            if not np.isfinite(s) or s <= 0.0:
                raise FloatingPointError(f"zero calibrated marginal mass for node {v}: {m}")
            marg[v] = float(m[1] / s)
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
