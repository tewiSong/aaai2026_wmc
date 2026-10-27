"""Inference engines for is-a implication theories with Bernoulli literal weights.

Target: p_T(y) proportional to prod_i p_i^{y_i} (1-p_i)^{1-y_i} * [y_c <= y_p for every edge].

Engines
  exact_marginals_brute        enumeration, n <= ~22 (reference for tests)
  exact_marginals_ve_query     dense variable elimination, conditioning queries (low width)
  Circuit / compile_with_d4    d4 decision-DNNF compiled once per theory; every score vector
                               is then evaluated in time linear in the circuit (all marginals
                               by one upward + one downward pass, log space)
  loopy_bp                     synchronous damped sum-product, log space
  gibbs_marginals              chromatic heat-bath Gibbs, many chains, half started from the
                               all-zeros and half from the all-ones state (monotone bracket)
"""
from __future__ import annotations

import itertools
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit, logit

from wmc_common import EXP_DIR

D4_BIN = EXP_DIR / "tools" / "bin" / "d4"
CLIP = 1e-6

# --------------------------------------------------------------------------- scores


def make_scores(n: int, family: str, rng: np.random.Generator) -> np.ndarray:
    """Synthetic score families. 'sparse' is a predictor-like proxy (most terms near 0,
    a few high) used until real DiamondScore vectors are available."""
    if family == "uniform":
        p = rng.uniform(0, 1, n)
    elif family == "beta22":
        p = rng.beta(2, 2, n)
    elif family == "beta0505":
        p = rng.beta(0.5, 0.5, n)
    elif family == "sparse":
        hot = rng.random(n) < 0.03
        p = np.where(hot, rng.beta(5, 2, n), rng.beta(1, 50, n))
    else:
        raise ValueError(family)
    return np.clip(p, CLIP, 1 - CLIP)


# --------------------------------------------------------------------------- references


def exact_marginals_brute(n: int, edges: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, float]:
    ys = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int8)
    ok = np.ones(len(ys), dtype=bool)
    for c, q in edges:
        ok &= ~((ys[:, c] == 1) & (ys[:, q] == 0))
    ys = ys[ok]
    w = np.prod(np.where(ys == 1, p, 1 - p), axis=1)
    z = w.sum()
    return (w[:, None] * ys).sum(0) / z, float(z)


def ve_log_z(n: int, edges: np.ndarray, p: np.ndarray, order: list[int],
             clamp: dict[int, int] | None = None) -> float:
    """Dense variable elimination along `order`; returns log Z (with clamped variables)."""
    clamp = clamp or {}
    factors: list[tuple[tuple[int, ...], np.ndarray]] = []
    for i in range(n):
        t = np.array([1 - p[i], p[i]])
        if i in clamp:
            t = t * (np.arange(2) == clamp[i])
        factors.append(((i,), t))
    imp = np.array([[1.0, 1.0], [0.0, 1.0]])  # [y_c, y_p]
    for c, q in edges:
        factors.append(((int(c), int(q)), imp))
    log_scale = 0.0
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    bucket: dict[int, list[int]] = {v: [] for v in range(n)}
    alive = [True] * len(factors)
    pos = {v: k for k, v in enumerate(order)}
    for fi, (sc, _) in enumerate(factors):
        bucket[min(sc, key=pos.get)].append(fi)
    for v in order:
        fis = [fi for fi in bucket[v] if alive[fi]]
        if not fis:
            continue
        scope = sorted({u for fi in fis for u in factors[fi][0]}, key=pos.get)
        sym = {u: letters[k] for k, u in enumerate(scope)}
        out = [u for u in scope if u != v]
        full = "".join(sym[u] for u in scope)
        acc = np.ones((2,) * len(scope))
        for fi in fis:  # pairwise products: a hub bucket can hold hundreds of factors
            acc = np.einsum(f"{full},{''.join(sym[u] for u in factors[fi][0])}->{full}",
                            acc, factors[fi][1])
        val = acc.sum(axis=scope.index(v))
        for fi in fis:
            alive[fi] = False
        m = val.max()
        if m <= 0:
            return -np.inf
        val = val / m
        log_scale += np.log(m)
        if out:
            factors.append((tuple(out), val))
            alive.append(True)
            bucket[min(out, key=pos.get)].append(len(factors) - 1)
    return log_scale


# --------------------------------------------------------------------------- d4 circuit


def compile_with_d4(cnf: Path, nnf_out: Path, timeout_s: float | None = None) -> dict:
    t0 = time.time()
    res = subprocess.run([str(D4_BIN), "-dDNNF", str(cnf), f"-out={nnf_out}"],
                         capture_output=True, text=True, timeout=timeout_s)
    if res.returncode not in (0, 10, 20) or not nnf_out.exists():
        raise RuntimeError(f"d4 failed rc={res.returncode}: {res.stdout[-2000:]} {res.stderr[-2000:]}")
    count = next((l.split()[1] for l in res.stdout.splitlines() if l.startswith("s ")), None)
    return dict(seconds=round(time.time() - t0, 3), model_count=count)


class Circuit:
    """Decision-DNNF from d4. Node types: 'o' (deterministic or), 'a' (decomposable and),
    't', 'f'. Edges carry the literals set on that branch. Variables never mentioned on a
    model's path are free; their weight factor p + (1 - p) = 1 needs no smoothing."""

    def __init__(self, nnf: Path, n_vars: int):
        ids, types, par, chi, lit_edge, lits = [], [], [], [], [], []
        with open(nnf) as fh:
            for line in fh:
                tok = line.split()
                if not tok or tok[0] == "c":
                    continue
                if tok[0] in ("o", "a", "t", "f"):
                    ids.append(int(tok[1]))
                    types.append(tok[0])
                else:
                    e = len(par)
                    par.append(int(tok[0]))
                    chi.append(int(tok[1]))
                    for t in tok[2:-1]:
                        lit_edge.append(e)
                        lits.append(int(t))
        remap = {old: k for k, old in enumerate(ids)}
        self.n_vars = n_vars
        self.n_nodes = len(ids)
        self.types = np.array([{"o": 0, "a": 1, "t": 2, "f": 3}[t] for t in types], dtype=np.int8)
        par = np.array([remap[x] for x in par], dtype=np.int64)
        chi = np.array([remap[x] for x in chi], dtype=np.int64)
        order = np.argsort(par, kind="stable")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        self.par, self.chi = par[order], chi[order]
        self.lit_edge = inv[np.array(lit_edge, dtype=np.int64)] if lit_edge else np.zeros(0, np.int64)
        lits = np.array(lits, dtype=np.int64)
        self.lit_var = np.abs(lits) - 1
        self.lit_pos = lits > 0
        self.n_edges = len(self.par)

        indeg = np.bincount(self.chi, minlength=self.n_nodes)
        roots = np.flatnonzero(indeg == 0)
        if len(roots) != 1:
            raise ValueError(f"expected one root, found {len(roots)}")
        self.root = int(roots[0])

        # height (children first) and depth (parents first) levels for vectorised passes
        start = np.searchsorted(self.par, np.arange(self.n_nodes + 1))
        self.edge_start = start
        topo, indeg_w, stack = [], indeg.copy(), [self.root]
        while stack:
            u = stack.pop()
            topo.append(u)
            for e in range(start[u], start[u + 1]):
                c = self.chi[e]
                indeg_w[c] -= 1
                if indeg_w[c] == 0:
                    stack.append(c)
        if len(topo) != self.n_nodes:
            raise ValueError("circuit has unreachable nodes or a cycle")
        topo = np.array(topo)
        height = np.zeros(self.n_nodes, dtype=np.int64)
        for u in topo[::-1]:
            s, t = start[u], start[u + 1]
            if t > s:
                height[u] = height[self.chi[s:t]].max() + 1
        depth = np.zeros(self.n_nodes, dtype=np.int64)
        for u in topo:
            s, t = start[u], start[u + 1]
            if t > s:
                np.maximum.at(depth, self.chi[s:t], depth[u] + 1)
        self.height, self.depth = height, depth
        self.edge_height = height[self.par]
        self.edge_depth = depth[self.par]

    def _edge_logw(self, logp, log1mp):
        lw = np.where(self.lit_pos, logp[self.lit_var], log1mp[self.lit_var])
        return np.bincount(self.lit_edge, weights=lw, minlength=self.n_edges)

    def marginals(self, p: np.ndarray) -> tuple[np.ndarray, float]:
        return self._marginals(np.log(p), np.log1p(-p), p)

    def marginals_theta(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        """Same as marginals(sigmoid(theta)) but computed from log-odds without clipping:
        log p = -softplus(-theta), log(1-p) = -softplus(theta). Exact for any |theta|
        (fix for the clipping inconsistency found in the Codex P4.4-P4.6 audit)."""
        return self._marginals(-np.logaddexp(0.0, -theta), -np.logaddexp(0.0, theta), expit(theta))

    def _marginals(self, logp: np.ndarray, log1mp: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, float]:
        ew = self._edge_logw(logp, log1mp)
        val = np.full(self.n_nodes, -np.inf)
        val[self.types == 2] = 0.0
        ev = np.full(self.n_edges, -np.inf)
        for h in range(1, self.height.max() + 1):
            es = np.flatnonzero(self.edge_height == h)
            ev[es] = val[self.chi[es]] + ew[es]
            nodes = np.unique(self.par[es])
            seg = np.searchsorted(self.par[es], nodes)
            is_or = self.types[nodes] == 0
            m = np.maximum.reduceat(ev[es], seg)
            m_safe = np.where(np.isfinite(m), m, 0.0)
            lse = m_safe + np.log(np.add.reduceat(np.exp(ev[es] - np.repeat(m_safe, np.diff(np.append(seg, len(es))))), seg))
            lse = np.where(np.isfinite(m), lse, -np.inf)
            sm = np.add.reduceat(ev[es], seg)
            val[nodes] = np.where(is_or, lse, sm)
        logz = val[self.root]

        out = np.full(self.n_nodes, -np.inf)
        out[self.root] = 0.0
        mass = np.full(self.n_edges, -np.inf)  # log mass of models passing through edge
        for d in range(0, self.depth.max() + 1):
            es = np.flatnonzero(self.edge_depth == d)
            if len(es) == 0:
                continue
            pa = self.par[es]
            is_or = self.types[pa] == 0
            with np.errstate(invalid="ignore"):
                to_child = np.where(is_or, out[pa] + ew[es], out[pa] + val[pa] - ev[es] + ew[es])
            to_child = np.where(np.isfinite(val[pa]) & np.isfinite(ev[es]), to_child, -np.inf)
            mass[es] = np.where(is_or, out[pa] + ev[es], out[pa] + val[pa])
            np.logaddexp.at(out, self.chi[es], to_child)

        rel = np.exp(mass - logz)  # fraction of posterior mass through each edge
        rel_l = rel[self.lit_edge]
        decided = np.bincount(self.lit_var, weights=rel_l, minlength=self.n_vars)
        decided_true = np.bincount(self.lit_var, weights=np.where(self.lit_pos, rel_l, 0.0),
                                   minlength=self.n_vars)
        mu = decided_true + p * (1.0 - decided)
        return np.clip(mu, 0.0, 1.0), float(logz)


# --------------------------------------------------------------------------- loopy BP


def loopy_bp(n: int, edges: np.ndarray, p: np.ndarray, damping: float = 0.5,
             tol: float = 1e-13, max_iter: int = 20000) -> dict:
    # tol was 1e-9 for P1.0a/P1.2; with scores clipped at 1e-9 that stopped too early on
    # GO mf P24387 (error 0.028 -> 0 at 1e-13). GO bp results were unchanged (Codex audit).
    c, q = edges[:, 0], edges[:, 1]
    lphi = np.stack([np.log1p(-p), np.log(p)], axis=1)
    m_up = np.zeros((len(edges), 2))    # child -> parent, over y_parent (log, normalised)
    m_dn = np.zeros((len(edges), 2))    # parent -> child, over y_child
    m_up -= np.log(2)
    m_dn -= np.log(2)
    it, delta = 0, np.inf
    for it in range(1, max_iter + 1):
        lb = lphi.copy()
        np.add.at(lb, q, m_up)
        np.add.at(lb, c, m_dn)
        cav_c = lb[c] - m_dn          # child's cavity w.r.t. this edge
        cav_p = lb[q] - m_up
        new_up = np.stack([cav_c[:, 0], np.logaddexp(cav_c[:, 0], cav_c[:, 1])], axis=1)
        new_dn = np.stack([np.logaddexp(cav_p[:, 0], cav_p[:, 1]), cav_p[:, 1]], axis=1)
        new_up -= np.logaddexp(new_up[:, :1], new_up[:, 1:])
        new_dn -= np.logaddexp(new_dn[:, :1], new_dn[:, 1:])
        old_up, old_dn = np.exp(m_up), np.exp(m_dn)
        up = damping * old_up + (1 - damping) * np.exp(new_up)
        dn = damping * old_dn + (1 - damping) * np.exp(new_dn)
        delta = max(np.abs(up - old_up).max(), np.abs(dn - old_dn).max()) if len(edges) else 0.0
        m_up, m_dn = np.log(up), np.log(dn)
        if delta < tol:
            break
    lb = lphi.copy()
    np.add.at(lb, q, m_up)
    np.add.at(lb, c, m_dn)
    mu = expit(lb[:, 1] - lb[:, 0])
    return dict(mu=mu, iterations=it, converged=bool(delta < tol), final_delta=float(delta))


# --------------------------------------------------------------------------- Gibbs


def greedy_coloring(n: int, edges: np.ndarray) -> np.ndarray:
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(int(b))
        adj[b].append(int(a))
    color = np.full(n, -1, dtype=np.int64)
    for v in np.argsort([-len(a) for a in adj], kind="stable"):
        used = {color[u] for u in adj[v] if color[u] >= 0}
        k = 0
        while k in used:
            k += 1
        color[v] = k
    return color


def gibbs_marginals(n: int, edges: np.ndarray, p: np.ndarray, rng: np.random.Generator,
                    chains: int = 32, burn_in: int = 500, sweeps: int = 5000) -> dict:
    """Chromatic heat-bath Gibbs. Under y_c <= y_p the conditional of y_i is: 1 if any child
    is 1, 0 if any parent is 0, otherwise Bernoulli(p_i). Half of the chains start at all
    zeros, half at all ones (both valid). The chains use independent random numbers, so this
    is NOT a coupled (CFTP-style) bracket: the gap between the two halves and the chain
    standard error are empirical mixing diagnostics, not a proven confidence bound."""
    c, q = edges[:, 0], edges[:, 1]
    ones = np.ones(len(edges), dtype=np.float32)
    children = csr_matrix((ones, (q, c)), shape=(n, n))  # row parent -> its children
    parents = csr_matrix((ones, (c, q)), shape=(n, n))   # row child -> its parents
    color = greedy_coloring(n, edges)
    classes = [np.flatnonzero(color == k) for k in range(color.max() + 1)]
    sub = [(idx, children[idx], parents[idx]) for idx in classes]
    y = np.zeros((n, chains), dtype=np.float32)
    y[:, chains // 2:] = 1.0
    acc = np.zeros((n, chains))
    for s in range(burn_in + sweeps):
        for idx, ch, pa in sub:
            any_child_on = (ch @ y) > 0
            any_parent_off = (pa @ (1.0 - y)) > 0
            draw = rng.random((len(idx), chains)) < p[idx, None]
            y[idx] = np.where(any_child_on, 1.0, np.where(any_parent_off, 0.0, draw))
        if s >= burn_in:
            acc += y
    per_chain = acc / sweeps
    mu = per_chain.mean(1)
    lo, hi = per_chain[:, : chains // 2].mean(1), per_chain[:, chains // 2:].mean(1)
    se = per_chain.std(1, ddof=1) / np.sqrt(chains)
    return dict(mu=mu, bracket_gap=np.abs(hi - lo), chain_se=se, colors=len(classes))


def violations(edges: np.ndarray, mu: np.ndarray, tol: float = 1e-9) -> int:
    """Number of edges with marginal(child) > marginal(parent); impossible under p_T."""
    return int(np.sum(mu[edges[:, 0]] > mu[edges[:, 1]] + tol))


# --------------------------------------------------------------------------- certified pruning


def descendant_max_score(n: int, edges: np.ndarray, p: np.ndarray) -> np.ndarray:
    """m(v) = max score over v and all its descendants (children are edge sources)."""
    children = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=np.int64)  # number of children not yet processed
    parents = [[] for _ in range(n)]
    for c, q in edges:
        children[q].append(int(c))
        parents[c].append(int(q))
        indeg[q] += 1
    m = p.astype(float).copy()
    stack = [v for v in range(n) if indeg[v] == 0]  # leaves first
    while stack:
        v = stack.pop()
        for u in parents[v]:
            if m[v] > m[u]:
                m[u] = m[v]
            indeg[u] -= 1
            if indeg[u] == 0:
                stack.append(u)
    return m


def cold_set_for_budget(n: int, edges: np.ndarray, p: np.ndarray, delta_target: float):
    """Largest down-closed cold set D = {v : m(v) <= tau} whose certificate
    delta = 1 - prod_{d in D} (1 - p_d) stays <= delta_target. Returns (in_D mask, delta, tau)."""
    m = descendant_max_score(n, edges, p)
    order = np.argsort(m, kind="stable")
    cum = np.cumsum(np.log1p(-p[order]))           # log prod (1 - p) over the prefix
    ok = -np.expm1(cum) <= delta_target
    # thresholds must not split ties in m (D must be {m <= tau})
    last = -1
    for k in np.flatnonzero(ok)[::-1]:
        if k == n - 1 or m[order[k]] < m[order[k + 1]]:
            last = k
            break
    in_d = np.zeros(n, dtype=bool)
    if last >= 0:
        in_d[order[: last + 1]] = True
    delta = float(-np.expm1(np.log1p(-p[in_d]).sum())) if in_d.any() else 0.0
    tau = float(m[order[last]]) if last >= 0 else -1.0
    return in_d, delta, tau


def certified_marginals(n: int, edges: np.ndarray, p: np.ndarray, delta_target: float,
                        workdir: Path, d4_timeout: float | None = None) -> dict:
    """Prune a down-closed cold set D (y_D = 0 is always consistent, and the pruned model is
    the exact posterior conditioned on y_D = 0), solve the remaining active core exactly with
    d4, and return marginals with certified intervals:
        P(y_D = 0) >= prod_{d in D} (1 - p_d) = 1 - delta, hence for every term
        r in core: mu_r in [(1 - delta) mu_hat_r, mu_hat_r + delta (1 - mu_hat_r)]
        d in D   : mu_d in [0, delta]
    The same TV bound delta holds for every Boolean query."""
    from wmc_common import write_dimacs
    in_d, delta, tau = cold_set_for_budget(n, edges, p, delta_target)
    core = np.flatnonzero(~in_d)
    mu_hat = np.zeros(n)
    info = dict(core_size=int(len(core)), delta=delta, tau=tau)
    if len(core):
        idx = -np.ones(n, dtype=np.int64)
        idx[core] = np.arange(len(core))
        keep = (idx[edges[:, 0]] >= 0) & (idx[edges[:, 1]] >= 0)
        sub = idx[edges[keep]].astype(np.int32)
        info["core_edges"] = int(len(sub))
        t0 = time.time()
        if len(sub):
            cnf, nnf = workdir / "core.cnf", workdir / "core.nnf"
            write_dimacs(cnf, len(core), sub)
            compile_with_d4(cnf, nnf, timeout_s=d4_timeout)
            mu_core, _ = Circuit(nnf, len(core)).marginals(p[core])
        else:
            mu_core = p[core].copy()
        info["core_seconds"] = round(time.time() - t0, 3)
        mu_hat[core] = mu_core
    lo = np.where(in_d, 0.0, (1 - delta) * mu_hat)
    hi = np.where(in_d, delta, mu_hat + delta * (1 - mu_hat))
    return dict(mu_hat=mu_hat, lo=lo, hi=hi, **info)


def loopy_bp_theta(n: int, edges: np.ndarray, theta: np.ndarray, init=None, damping: float = 0.5,
                   tol: float = 1e-13, max_iter: int = 20000) -> dict:
    """Loopy BP for P(y) ~ exp(theta . y)[y valid], from log-odds (no clipping), with optional
    warm-start messages (init = (m_up, m_dn) in log space). Same updates as loopy_bp."""
    c, q = edges[:, 0], edges[:, 1]
    lphi = np.stack([-np.logaddexp(0.0, theta), -np.logaddexp(0.0, -theta)], axis=1)
    if init is None:
        m_up = np.full((len(edges), 2), -np.log(2)); m_dn = m_up.copy()
    else:
        m_up, m_dn = init[0].copy(), init[1].copy()
    it, delta = 0, np.inf
    for it in range(1, max_iter + 1):
        lb = lphi.copy()
        np.add.at(lb, q, m_up); np.add.at(lb, c, m_dn)
        cav_c, cav_p = lb[c] - m_dn, lb[q] - m_up
        new_up = np.stack([cav_c[:, 0], np.logaddexp(cav_c[:, 0], cav_c[:, 1])], axis=1)
        new_dn = np.stack([np.logaddexp(cav_p[:, 0], cav_p[:, 1]), cav_p[:, 1]], axis=1)
        new_up -= np.logaddexp(new_up[:, :1], new_up[:, 1:])
        new_dn -= np.logaddexp(new_dn[:, :1], new_dn[:, 1:])
        up = damping * np.exp(m_up) + (1 - damping) * np.exp(new_up)
        dn = damping * np.exp(m_dn) + (1 - damping) * np.exp(new_dn)
        delta = max(np.abs(up - np.exp(m_up)).max(), np.abs(dn - np.exp(m_dn)).max()) if len(edges) else 0.0
        m_up, m_dn = np.log(np.maximum(up, 1e-300)), np.log(np.maximum(dn, 1e-300))
        if delta < tol:
            break
    lb = lphi.copy()
    np.add.at(lb, q, m_up); np.add.at(lb, c, m_dn)
    return dict(mu=expit(lb[:, 1] - lb[:, 0]), messages=(m_up, m_dn), iterations=it, converged=bool(delta < tol))


def loopy_bp_logodds(n: int, edges: np.ndarray, theta: np.ndarray, init=None, damping: float = 0.5,
                     tol: float = 1e-10, max_iter: int = 20000) -> dict:
    """Loopy BP with scalar log-ratio messages (fix from the Codex P4.7 audit). For c => p:
    child->parent r_up = softplus(cavity log-odds of c), parent->child r_dn = -softplus(-cavity
    log-odds of p). Damping and the stopping test act on log-ratios, so near-deterministic messages
    are neither floored nor declared converged early. Same fixed-point equations as loopy_bp."""
    c, q = edges[:, 0], edges[:, 1]
    E = len(edges)
    r_up, r_dn = (np.zeros(E), np.zeros(E)) if init is None else (init[0].copy(), init[1].copy())
    it, resid = 0, np.inf
    for it in range(1, max_iter + 1):
        b = theta.astype(float).copy()
        np.add.at(b, q, r_up); np.add.at(b, c, r_dn)
        new_up = np.logaddexp(0.0, b[c] - r_dn)
        new_dn = -np.logaddexp(0.0, -(b[q] - r_up))
        resid = max(np.abs(new_up - r_up).max(), np.abs(new_dn - r_dn).max()) if E else 0.0
        r_up = damping * r_up + (1 - damping) * new_up
        r_dn = damping * r_dn + (1 - damping) * new_dn
        if resid < tol:
            break
    b = theta.astype(float).copy()
    np.add.at(b, q, r_up); np.add.at(b, c, r_dn)
    return dict(mu=expit(b), messages=(r_up, r_dn), iterations=it, converged=bool(resid < tol), residual=float(resid))


# --------------------------------------------------------------------------- hot-core exact + periphery BP (P6)


class HotCoreHybrid:
    """Hybrid inference for one protein (plan item (d), P6.1). The terms split into an ancestor-
    closed hot core H (e.g. {mu*_i >= tau} of the isotonic target) and its descendant-closed cold
    complement. The core is ONE region solved exactly by a d4 circuit; every edge with a cold child
    carries log-odds BP messages (as loopy_bp_logodds). A message r_up from a cold child enters the
    core as a unary log-odds term on its hot parent, so the core's cavity for that edge,
    logit(mu_p) - r_up, is exact for the region. Only loops through the cold periphery are
    approximated; with H = all terms this is exact inference, with H = {} it is loopy BP."""

    def __init__(self, n: int, edges: np.ndarray, core_mask: np.ndarray, workdir: Path, d4_timeout=None):
        from wmc_common import write_dimacs
        self.n = n
        self.H = np.flatnonzero(core_mask)
        self.C = np.flatnonzero(~core_mask)
        hidx = -np.ones(n, dtype=np.int64)
        hidx[self.H] = np.arange(len(self.H))
        inner = core_mask[edges[:, 0]] & core_mask[edges[:, 1]]
        if np.any(core_mask[edges[:, 0]] & ~core_mask[edges[:, 1]]):
            raise ValueError("core is not ancestor-closed")
        self.circ = None
        if inner.any():
            cnf, nnf = Path(workdir) / "core.cnf", Path(workdir) / "core.nnf"
            write_dimacs(cnf, len(self.H), hidx[edges[inner]])
            compile_with_d4(cnf, nnf, timeout_s=d4_timeout)
            self.circ = Circuit(nnf, len(self.H))
        pe = edges[~inner]
        self.ec, self.ep = pe[:, 0], pe[:, 1]
        self.hot_parent = core_mask[self.ep]
        self.hp_local = hidx[self.ep[self.hot_parent]]
        self.E = len(pe)

    def _core(self, th_h):
        if self.circ is None:
            return expit(th_h)
        return self.circ.marginals_theta(th_h)[0]

    def _incoming(self, r_up, r_dn):
        inc = np.zeros(self.n)
        np.add.at(inc, self.ep, r_up)
        np.add.at(inc, self.ec, r_dn)
        return inc

    def marginals(self, theta: np.ndarray, init=None, damping: float = 0.5, tol: float = 1e-9,
                  max_outer: int = 500, inner: int = 5) -> dict:
        # Near-deterministic periphery messages have several fixed points (P6.1 self-test: from
        # zero messages a cold term with large theta switches its parents on). init="low" starts
        # from "every parent off" and moves up; pass the moment-matched messages to warm-start.
        if init is None or (isinstance(init, str) and init == "low"):
            r_up, r_dn = np.zeros(self.E), np.full(self.E, -50.0)
        else:
            r_up, r_dn = init[0].copy(), init[1].copy()
        mu = np.zeros(self.n)
        resid, it = np.inf, 0
        for it in range(1, max_outer + 1):
            th_h = theta[self.H].astype(float).copy()
            np.add.at(th_h, self.hp_local, r_up[self.hot_parent])
            mu_h = np.clip(self._core(th_h), 1e-300, 1 - 1e-16)
            b_hot = np.zeros(self.n)
            b_hot[self.H] = logit(mu_h)
            cav_hot = b_hot[self.ep] - r_up                      # exact region cavity, frozen for the inner sweeps
            start_up, start_dn = r_up.copy(), r_dn.copy()
            for _ in range(inner):
                b = theta.astype(float) + self._incoming(r_up, r_dn)
                cav_p = np.where(self.hot_parent, cav_hot, b[self.ep] - r_up)
                new_up = np.logaddexp(0.0, b[self.ec] - r_dn)
                new_dn = -np.logaddexp(0.0, -cav_p)
                r_up = damping * r_up + (1 - damping) * new_up
                r_dn = damping * r_dn + (1 - damping) * new_dn
            resid = max(np.abs(r_up - start_up).max(), np.abs(r_dn - start_dn).max()) if self.E else 0.0
            mu[self.H] = mu_h
            if resid < tol:
                break
        b = theta.astype(float) + self._incoming(r_up, r_dn)
        mu[self.C] = expit(b[self.C])
        th_h = theta[self.H].astype(float).copy()
        np.add.at(th_h, self.hp_local, r_up[self.hot_parent])
        mu[self.H] = self._core(th_h)
        return dict(mu=mu, messages=(r_up, r_dn), iterations=it, converged=bool(resid < tol), residual=float(resid))

    def moment_match(self, mu_t: np.ndarray, damping: float = 0.5, tol: float = 1e-10, max_iter: int = 50000) -> dict:
        """theta whose hybrid marginals equal mu_t: pin every belief to logit(mu_t), iterate the
        periphery messages to their fixed point, set cold theta = logit(mu_t) - incoming, and match
        the core exactly (L-BFGS on the core circuit) with the cold-child messages as unaries."""
        from learned_constrained import moment_match as mm_exact
        b = logit(np.clip(mu_t, 1e-15, 1 - 1e-15))
        r_up, r_dn = np.zeros(self.E), np.zeros(self.E)
        resid, it = 0.0, 0
        for it in range(1, max_iter + 1):
            if not self.E:
                break
            new_up = np.logaddexp(0.0, b[self.ec] - r_dn)
            new_dn = -np.logaddexp(0.0, -(b[self.ep] - r_up))
            resid = max(np.abs(new_up - r_up).max(), np.abs(new_dn - r_dn).max())
            r_up = damping * r_up + (1 - damping) * new_up
            r_dn = damping * r_dn + (1 - damping) * new_dn
            if resid < tol:
                break
        theta = b - self._incoming(r_up, r_dn)
        if self.circ is not None:
            th_core, _ = mm_exact(self.circ, mu_t[self.H], b[self.H])
        else:
            th_core = b[self.H]
        unary = np.zeros(len(self.H))
        np.add.at(unary, self.hp_local, r_up[self.hot_parent])
        theta[self.H] = th_core - unary
        return dict(theta=theta, messages=(r_up, r_dn), iterations=it, converged=bool(resid < tol), residual=float(resid))
