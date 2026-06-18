"""Soft closure of the true-path theory = loopy belief propagation.

The deployed soft closure runs message passing on the constraint factor graph of the
true-path theory: a binary variable per GO term, a unary factor carrying the prior
q_i, and one factor per Horn clause (NF1 / NF2 / NF3 / NF4) that is 1 on satisfying
local assignments and 0 otherwise. We run sum-product loopy belief propagation to a
fixpoint and read the per-variable beliefs.

This is "belief propagation" in the literal sense of the paper's central claim, and it
is exact on any tree fragment (the standard BP-on-trees result, which we verify on
chains and stars). On a graph with reconvergences it only approximates the exact WMC
marginal, with the error confined to the reconvergent neighbourhoods (a diamond gives
the reconvergent node ~0.85 / sink ~0.15 against the exact 5/6 and 1/6).

Damping stabilizes the iteration on loopy graphs without changing the fixpoint; the
schedule is synchronous. No clamping or approximation of the factors is used: every
factor is the exact Boolean indicator of its clause.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .graph import TruePathTheory


class _FactorGraph:
    """Factor graph for a clause set with unary priors, set up for sum-product BP."""

    def __init__(self, n_vars: int, clauses: Sequence[Tuple[int, ...]], priors: np.ndarray):
        self.n = n_vars
        self.priors = priors
        # Each clause factor: list of 0-based vars and a truth table over them.
        self.factor_vars: List[List[int]] = []
        self.factor_tab: List[np.ndarray] = []
        for clause in clauses:
            vs = sorted({abs(l) - 1 for l in clause})
            want = {abs(l) - 1: (1 if l > 0 else 0) for l in clause}
            tab = np.zeros((2,) * len(vs), dtype=np.float64)
            for idx in np.ndindex(*tab.shape):
                assign = {v: idx[k] for k, v in enumerate(vs)}
                # Clause satisfied if any literal matches.
                sat = any(assign[v] == want[v] for v in vs)
                tab[idx] = 1.0 if sat else 0.0
            self.factor_vars.append(vs)
            self.factor_tab.append(tab)
        # Incidence: factors touching each variable.
        self.var_factors: List[List[int]] = [[] for _ in range(n_vars)]
        for fi, vs in enumerate(self.factor_vars):
            for v in vs:
                self.var_factors[v].append(fi)


def soft_closure_bp(
    theory: TruePathTheory,
    priors: Dict[str, float],
    max_iter: int = 5000,
    tol: float = 1e-10,
    damping: float = 0.5,
) -> Dict[str, float]:
    """Loopy sum-product BP marginals for the whole theory.

    Returns per-atom belief P(y=1). Exact on tree fragments; approximate at
    reconvergences. `damping` mixes new and old factor->var messages for stability.
    """
    atoms = theory.atoms
    n = len(atoms)
    pri = np.array([priors[a] for a in atoms], dtype=np.float64)
    beliefs = _run_bp(n, theory.clauses(), pri, max_iter, tol, damping)
    return {a: float(beliefs[i]) for i, a in enumerate(atoms)}


def _run_bp(
    n: int,
    clauses: Sequence[Tuple[int, ...]],
    pri: np.ndarray,
    max_iter: int,
    tol: float,
    damping: float,
) -> np.ndarray:
    fg = _FactorGraph(n, clauses, pri)
    # Messages var->factor and factor->var, each a length-2 nonneg vector.
    m_vf: Dict[Tuple[int, int], np.ndarray] = {}
    m_fv: Dict[Tuple[int, int], np.ndarray] = {}
    for fi, vs in enumerate(fg.factor_vars):
        for v in vs:
            m_vf[(v, fi)] = np.array([1.0, 1.0])
            m_fv[(fi, v)] = np.array([1.0, 1.0])

    def normalize(x: np.ndarray) -> np.ndarray:
        s = x.sum()
        return x / s if s > 0 else np.array([0.5, 0.5])

    for _ in range(max_iter):
        max_delta = 0.0
        # var -> factor: product of incoming factor messages and the prior, excluding fi.
        for v in range(n):
            prior_msg = np.array([1.0 - pri[v], pri[v]])
            incoming = [m_fv[(fi, v)] for fi in fg.var_factors[v]]
            for k, fi in enumerate(fg.var_factors[v]):
                msg = prior_msg.copy()
                for j, other in enumerate(incoming):
                    if j != k:
                        msg = msg * other
                m_vf[(v, fi)] = normalize(msg)
        # factor -> var: marginalize the factor table against other vars' messages.
        for fi, vs in enumerate(fg.factor_vars):
            tab = fg.factor_tab[fi]
            for vi, v in enumerate(vs):
                # Contract tab over all other variables weighted by their var->factor msgs.
                t = tab
                # Multiply in messages for every other axis, then sum them out.
                msg_axes = []
                for k, u in enumerate(vs):
                    if u == v:
                        continue
                    shape = [1] * tab.ndim
                    shape[k] = 2
                    t = t * m_vf[(u, fi)].reshape(shape)
                # Sum out all axes except vi.
                axes = tuple(k for k in range(tab.ndim) if k != vi)
                out = t.sum(axis=axes) if axes else t
                out = normalize(np.asarray(out, dtype=np.float64).reshape(2))
                old = m_fv[(fi, v)]
                new = damping * old + (1.0 - damping) * out
                new = normalize(new)
                max_delta = max(max_delta, float(np.abs(new - old).max()))
                m_fv[(fi, v)] = new
        if max_delta < tol:
            break

    beliefs = np.zeros(n)
    for v in range(n):
        b = np.array([1.0 - pri[v], pri[v]])
        for fi in fg.var_factors[v]:
            b = b * m_fv[(fi, v)]
        b = normalize(b)
        beliefs[v] = b[1]
    return beliefs


def soft_closure_upward(
    theory: TruePathTheory,
    priors: Dict[str, float],
    tol: float = 1e-12,
    max_iter: int = 100000,
) -> Dict[str, float]:
    """Upward-only EL-completion soft-OR fixpoint (the paper's literal 'normalized soft-OR
    of sufficient conditions').

    mu_y = q_y / (q_y + (1-q_y) * prod_{x->y}(1-mu_x) * prod_{(a,b)->y}(1-mu_a mu_b)),
    iterated to its (monotone) fixpoint. This is exact on a star (Theorem 1) but, lacking
    the downward 'cap by the true-path constraint', it cannot lower a source/leaf below its
    prior -- so it is provided to *contrast* with the bidirectional closure (loopy BP),
    which is what the paper's empirical error figures reflect.
    """
    atoms = theory.atoms
    n = len(atoms)
    idx = {a: i for i, a in enumerate(atoms)}
    q = np.array([priors[a] for a in atoms], dtype=np.float64)
    unary = [[] for _ in range(n)]
    for x, y in theory.nf1:
        unary[idx[y]].append(idx[x])
    for x, y in theory.nf3:
        unary[idx[y]].append(idx[x])
    for x, y in theory.nf4:
        unary[idx[y]].append(idx[x])
    binary = [[] for _ in range(n)]
    for a, b, e in theory.nf2:
        binary[idx[e]].append((idx[a], idx[b]))
    mu = q.copy()
    for _ in range(max_iter):
        delta = 0.0
        new = mu.copy()
        for y in range(n):
            if not unary[y] and not binary[y]:
                continue
            prod = 1.0
            for x in unary[y]:
                prod *= (1.0 - mu[x])
            for a, b in binary[y]:
                prod *= (1.0 - mu[a] * mu[b])
            qy = q[y]
            denom = qy + (1.0 - qy) * prod
            val = qy / denom if denom > 0 else 1.0
            if val > new[y]:
                new[y] = val
            delta = max(delta, abs(new[y] - mu[y]))
        mu = new
        if delta < tol:
            break
    return {a: float(mu[i]) for i, a in enumerate(atoms)}


def star_softor(q_parent: float, q_children: Sequence[float]) -> float:
    """Closed-form soft-OR update for a star (Theorem 1 statement); exact on a star."""
    prod = 1.0
    for qc in q_children:
        prod *= (1.0 - qc)
    return q_parent / (q_parent + (1.0 - q_parent) * prod)
