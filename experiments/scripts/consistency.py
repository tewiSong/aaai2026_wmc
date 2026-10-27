"""Marginal-consistency baselines that need no WMC (Codex P4.0 audit; Obozinski et al. 2008).

With only implications y_c => y_p, a marginal vector is realisable by some distribution over
valid assignments iff 0 <= mu <= 1 and mu_c <= mu_p for every edge (order polytope; e.g.
Y_i = 1[U <= mu_i] with one uniform U). The closest consistent marginals in squared error are
therefore the isotonic regression of the scores on the DAG.

isotonic_dag solves  min_mu 1/2 ||mu - p||^2  s.t.  mu_c - mu_p <= 0  for many score vectors at
once, by FISTA on the dual (lambda >= 0 per edge, mu = p - A^T lambda). Values stay in [0, 1]
because isotonic solutions are averages of the inputs.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def isotonic_dag(P: np.ndarray, edges: np.ndarray, tol: float = 1e-10, max_iter: int = 200000):
    """P: (k, n) score vectors. Returns (mu (k, n), info)."""
    k, n = P.shape
    E = len(edges)
    if E == 0:
        return P.copy(), dict(iterations=0, max_violation=0.0)
    rows = np.repeat(np.arange(E), 2)
    cols = edges[:, [0, 1]].ravel()
    vals = np.tile([1.0, -1.0], E)
    A = csr_matrix((vals, (rows, cols)), shape=(E, n))
    AT = A.T.tocsr()
    AAT = (A @ AT).astype(float)
    if E <= 64:
        L = float(np.linalg.eigvalsh(AAT.toarray()).max()) * 1.01
    else:
        L = float(eigsh(AAT, k=1, which="LA", return_eigenvectors=False)[0]) * 1.01
    Pt = P.T                                   # (n, k)
    lam = np.zeros((E, k)); z = lam.copy(); t = 1.0
    it, viol = 0, np.inf
    for it in range(1, max_iter + 1):
        mu = Pt - AT @ z
        lam_new = np.maximum(0.0, z + (A @ mu) / L)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = lam_new + ((t - 1) / t_new) * (lam_new - lam)
        lam, t = lam_new, t_new
        if it % 50 == 0:
            mu = Pt - AT @ lam
            g = A @ mu                         # constraint values, want <= 0
            viol = float(max(g.max(), 0.0))
            slack = float(np.abs(lam * g).max())   # complementary slackness
            if viol <= tol and slack <= tol:
                break
    mu = (Pt - AT @ lam).T
    g = mu[:, edges[:, 0]] - mu[:, edges[:, 1]]
    return mu, dict(iterations=it, max_violation=float(max(g.max(), 0.0)))


def pav_chain(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators for a chain constraint y_0 >= y_1 >= ... (non-increasing)."""
    blocks = []  # [sum, count]
    for v in y:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    return np.concatenate([np.full(c, s / c) for s, c in blocks])


def isotonic_dag_hildreth(P: np.ndarray, edges: np.ndarray, tol: float = 1e-9, max_sweeps: int = 100000):
    """Same QP as isotonic_dag, solved by Hildreth's dual coordinate ascent: one exact
    projection per edge constraint per sweep, vectorised over the k score vectors."""
    mu = P.T.copy()                            # (n, k)
    E = len(edges)
    lam = np.zeros((E, P.shape[0]))
    cs, ps = edges[:, 0].astype(int), edges[:, 1].astype(int)
    sweep = 0
    for sweep in range(1, max_sweeps + 1):
        for e in range(E):
            c, q = cs[e], ps[e]
            new = np.maximum(0.0, lam[e] + 0.5 * (mu[c] - mu[q]))
            d = new - lam[e]
            if d.any():
                lam[e] = new
                mu[c] -= d
                mu[q] += d
        if sweep % 10 == 0:
            g = mu[cs] - mu[ps]
            if max(g.max(), 0.0) <= tol and np.abs(lam * g).max() <= tol:
                break
    g = mu[cs] - mu[ps]
    return mu.T, dict(iterations=sweep, max_violation=float(max(g.max(), 0.0)))


def enforce_order_exact(P: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Remove the <= tol residual violations of an approximate isotonic solution exactly:
    process terms parents-first and set mu_c = min(mu_c, mu_p) for every parent p. Changes values
    only by the solver tolerance; afterwards mu_c <= mu_p holds exactly for every edge."""
    n = P.shape[1]
    parents = [[] for _ in range(n)]
    children = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=int)
    for c, q in edges:
        parents[c].append(int(q)); children[q].append(int(c)); indeg[c] += 1
    order, stack = [], [v for v in range(n) if indeg[v] == 0]
    while stack:
        v = stack.pop(); order.append(v)
        for u in children[v]:
            indeg[u] -= 1
            if indeg[u] == 0:
                stack.append(u)
    out = P.copy()
    for v in order:
        for q in parents[v]:
            np.minimum(out[:, v], out[:, q], out=out[:, v])
    return out
