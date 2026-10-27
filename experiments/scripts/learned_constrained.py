"""Learned constrained model C and its non-WMC baselines (plan section 7.11).

Model C: P_theta(y | s) proportional to exp(sum_i theta_i y_i) [y valid], with
    theta_i = a_{d(i)} * logit(s_i) + b_{d(i)}      (parameters shared per term depth d)
Its marginals are computed exactly by a d4 circuit. The conditional log-likelihood of an observed
(valid) label vector y is  sum_i [y_i log p_i + (1-y_i) log(1-p_i)] - log Z_circuit(p),
p = sigmoid(theta), and its gradient w.r.t. theta_i is y_i - mu_i(theta).

Baselines share the same per-depth parametrisation and are fitted by independent per-label
log-loss, then projected onto the order polytope by isotonic regression:
  depth_platt   q_i = sigmoid(a_d logit s_i + b_d)
  term_platt    q_i = sigmoid(a_i logit s_i + b_i)  (ridge towards identity; few-positive terms)
  neighbor_lr   q_i = sigmoid(w_d . [1, l_i, max l_parents, min l_parents, max l_children])
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

EPS = 1e-6


def term_depth(n, edges, root):
    children = [[] for _ in range(n)]
    for c, q in edges:
        children[q].append(int(c))
    depth = np.full(n, -1)
    depth[root] = 0
    dq = deque([root])
    while dq:
        u = dq.popleft()
        for v in children[u]:
            if depth[v] < 0:
                depth[v] = depth[u] + 1
                dq.append(v)
    depth[depth < 0] = depth.max() + 1   # unreachable from root (should not happen)
    return depth


class ModelC:
    def __init__(self, circuit, depth, ridge=1e-3):
        self.c, self.depth, self.ridge = circuit, depth, ridge
        self.D = int(depth.max()) + 1

    def theta(self, params, L):
        a, b = params[: self.D], params[self.D:]
        return a[self.depth] * L + b[self.depth]

    def nll_grad(self, params, L, Y):
        """Mean negative conditional log-likelihood over proteins, and its gradient."""
        D, tot, ga, gb = self.D, 0.0, np.zeros(self.D), np.zeros(self.D)
        for l, y in zip(L, Y):
            th = self.a_b_theta(params, l)
            p = np.clip(expit(th), EPS, 1 - EPS)
            mu, logz = self.c.marginals(p)
            tot -= float(np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)) - logz)
            r = y - mu                                   # d loglik / d theta
            ga -= np.bincount(self.depth, weights=r * l, minlength=D)
            gb -= np.bincount(self.depth, weights=r, minlength=D)
        k = len(L)
        reg = params - np.r_[np.ones(D), np.zeros(D)]
        return tot / k + self.ridge * np.sum(reg ** 2), np.r_[ga, gb] / k + 2 * self.ridge * reg

    def a_b_theta(self, params, l):
        return self.theta(params, l)

    def fit(self, L, Y, maxiter=200):
        x0 = np.r_[np.ones(self.D), np.zeros(self.D)]
        res = minimize(self.nll_grad, x0, args=(L, Y), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=maxiter))
        self.params = res.x
        return res

    def predict(self, L):
        return np.stack([self.c.marginals(np.clip(expit(self.theta(self.params, l)), EPS, 1 - EPS))[0] for l in L])


def _logloss_fit(X, y, x0, ridge, target):
    def f(w):
        z = X @ w
        pr = np.clip(expit(z), EPS, 1 - EPS)
        loss = -np.mean(y * np.log(pr) + (1 - y) * np.log1p(-pr)) + ridge * np.sum((w - target) ** 2)
        g = X.T @ (pr - y) / len(y) + 2 * ridge * (w - target)
        return loss, g
    return minimize(f, x0, jac=True, method="L-BFGS-B").x


def fit_depth_platt(L, Y, depth, ridge=1e-3):
    D = int(depth.max()) + 1
    W = np.tile([1.0, 0.0], (D, 1))
    for d in range(D):
        m = depth == d
        if not m.any():
            continue
        l, y = L[:, m].ravel(), Y[:, m].ravel()
        X = np.c_[l, np.ones_like(l)]
        W[d] = _logloss_fit(X, y, W[d], ridge, np.array([1.0, 0.0]))
    return W


def apply_depth_platt(W, L, depth):
    return expit(W[depth, 0] * L + W[depth, 1])


def fit_term_platt(L, Y, ridge=1e-2, min_pos=3):
    n = L.shape[1]
    W = np.tile([1.0, 0.0], (n, 1))
    for i in range(n):
        if Y[:, i].sum() < min_pos:
            continue
        X = np.c_[L[:, i], np.ones(len(L))]
        W[i] = _logloss_fit(X, Y[:, i].astype(float), W[i], ridge, np.array([1.0, 0.0]))
    return W


def apply_term_platt(W, L):
    return expit(W[None, :, 0] * L + W[None, :, 1])


def neighbor_features(L, edges, n):
    parents = [[] for _ in range(n)]
    children = [[] for _ in range(n)]
    for c, q in edges:
        parents[c].append(int(q)); children[q].append(int(c))
    k = len(L)
    maxp, minp, maxc = L.copy(), L.copy(), np.full_like(L, logit(EPS))
    for i in range(n):
        if parents[i]:
            maxp[:, i] = L[:, parents[i]].max(1); minp[:, i] = L[:, parents[i]].min(1)
        if children[i]:
            maxc[:, i] = L[:, children[i]].max(1)
    return np.stack([np.ones_like(L), L, maxp, minp, maxc], axis=-1)   # k x n x 5


def fit_neighbor_lr(F, Y, depth, ridge=1e-3):
    D = int(depth.max()) + 1
    target = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    W = np.tile(target, (D, 1))
    for d in range(D):
        m = depth == d
        if not m.any():
            continue
        X = F[:, m].reshape(-1, 5); y = Y[:, m].ravel().astype(float)
        W[d] = _logloss_fit(X, y, W[d], ridge, target)
    return W


def apply_neighbor_lr(W, F, depth):
    return expit(np.einsum("kni,ni->kn", F, W[depth]))


# --------------------------------------------------------------------------- nested model C2


def log_partition(circ, theta):
    """log sum_valid exp(theta . y) and the marginals, computed stably from log-odds.
    (Earlier version clipped p before adding softplus(theta): objective and gradient disagreed
    for |theta| > ~27; e.g. single edge, theta = (-40, 40): true 0.693, old code 13.06.)"""
    mu, logz_c = circ.marginals_theta(theta)
    return logz_c + np.sum(np.logaddexp(0.0, theta)), mu


def moment_match(circ, mu_t, theta0, maxiter=5000):
    """theta* = argmin log Z(theta) - theta . mu_t; model marginals then equal mu_t (interior)."""
    def f(th):
        lz, mu = log_partition(circ, th)
        return lz - th @ mu_t, mu - mu_t
    res = minimize(f, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, gtol=1e-11, ftol=1e-15))
    _, mu = log_partition(circ, res.x)
    return res.x, float(np.abs(mu - mu_t).max())


class ModelC2:
    """Nested learned constrained model: theta_i = theta*_i + w_{d,0} + w_{d,1} l_i + w_{d,2} m_i,
    where theta* reproduces the (interior-shifted) isotonic marginals of the protein's scores,
    l = logit(score), m = logit(isotonic marginal). At w = 0 the model equals that isotonic
    target, so the family nests the isotonic baseline. Fitted by mean joint conditional NLL."""

    K = 3

    def __init__(self, circuit, depth, ridge=1e-3):
        self.c, self.depth, self.ridge = circuit, depth, ridge
        self.D = int(depth.max()) + 1
        self.w = np.zeros(self.D * self.K)

    def _theta(self, w, th0, l, m):
        W = w.reshape(self.D, self.K)[self.depth]
        return th0 + W[:, 0] + W[:, 1] * l + W[:, 2] * m

    def nll_grad(self, w, TH0, L, M, Y):
        tot, g = 0.0, np.zeros((self.D, self.K))
        for th0, l, m, y in zip(TH0, L, M, Y):
            th = self._theta(w, th0, l, m)
            lz, mu = log_partition(self.c, th)
            tot -= float(th @ y - lz)
            r = y - mu
            for k, f in enumerate((np.ones_like(l), l, m)):
                g[:, k] -= np.bincount(self.depth, weights=r * f, minlength=self.D)
        n = len(TH0)
        return tot / n + self.ridge * np.sum(w ** 2), g.ravel() / n + 2 * self.ridge * w

    def fit(self, TH0, L, M, Y, maxiter=300):
        res = minimize(self.nll_grad, np.zeros(self.D * self.K), args=(TH0, L, M, Y), jac=True,
                       method="L-BFGS-B", options=dict(maxiter=maxiter))
        self.w = res.x
        return res

    def predict(self, TH0, L, M, w=None):
        w = self.w if w is None else w
        return np.stack([log_partition(self.c, self._theta(w, th0, l, m))[1] for th0, l, m in zip(TH0, L, M)])


# --------------------------------------------------------------------------- C2 with BP (plan 7.14 b)


def moment_match_bp(n, edges, mu_t, theta0=None, step=0.5, tol=1e-7, max_outer=2000):
    """Find theta with BP marginals mu_BP(theta) = mu_t by damped logit-space fixed-point updates
    theta <- theta + step * (logit mu_t - logit mu_BP(theta)); BP messages are warm-started."""
    from wmc_inference import loopy_bp_theta
    lt = logit(np.clip(mu_t, 1e-12, 1 - 1e-12))
    th = lt.copy() if theta0 is None else theta0.copy()
    msgs, err = None, np.inf
    for k in range(1, max_outer + 1):
        bp = loopy_bp_theta(n, edges, th, init=msgs, tol=1e-12, max_iter=5000)
        msgs = bp["messages"]
        err = float(np.abs(bp["mu"] - mu_t).max())
        if err < tol:
            break
        th = th + step * (lt - logit(np.clip(bp["mu"], 1e-12, 1 - 1e-12)))
    return th, err, k


def train_c2_adam(marg_fn, TH0, L, M, Y, depth, steps=300, lr=0.05, ridge=1e-3, log_every=50):
    """Adam on C2's residual weights using only marginals (gradient = y - mu), so the same trainer
    serves exact-circuit and BP inference. marg_fn(k, theta) -> marginals of protein k."""
    D, K = int(depth.max()) + 1, ModelC2.K
    w = np.zeros(D * K); m1 = np.zeros_like(w); m2 = np.zeros_like(w)
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for t in range(1, steps + 1):
        g = np.zeros((D, K))
        W = w.reshape(D, K)[depth]
        for k in range(len(TH0)):
            th = TH0[k] + W[:, 0] + W[:, 1] * L[k] + W[:, 2] * M[k]
            r = Y[k] - marg_fn(k, th)
            for j, f in enumerate((np.ones_like(L[k]), L[k], M[k])):
                g[:, j] -= np.bincount(depth, weights=r * f, minlength=D)
        g = g.ravel() / len(TH0) + 2 * ridge * w
        m1 = b1 * m1 + (1 - b1) * g; m2 = b2 * m2 + (1 - b2) * g * g
        w = w - lr * (m1 / (1 - b1 ** t)) / (np.sqrt(m2 / (1 - b2 ** t)) + eps)
        if t % log_every == 0:
            hist.append(dict(step=t, grad_norm=float(np.linalg.norm(g))))
    return w, hist


def moment_match_bp_fast(n, edges, mu_t, damping=0.5, tol=1e-10, max_iter=50000):
    """Single-loop BP moment matching: at a BP fixed point the belief of term i is
    sigmoid(theta_i + incoming_i), incoming_i = sum of log-ratios of messages into i. Each
    iteration sets theta_i = logit(mu_t_i) - incoming_i and performs one damped message update;
    at convergence the BP beliefs equal mu_t exactly. Returns (theta, max belief error, iters)."""
    c, q = edges[:, 0], edges[:, 1]
    lt = logit(np.clip(mu_t, 1e-12, 1 - 1e-12))
    m_up = np.full((len(edges), 2), -np.log(2)); m_dn = m_up.copy()
    th = lt.copy()
    for it in range(1, max_iter + 1):
        inc = np.zeros(n)
        np.add.at(inc, q, m_up[:, 1] - m_up[:, 0]); np.add.at(inc, c, m_dn[:, 1] - m_dn[:, 0])
        th = lt - inc
        lphi = np.stack([-np.logaddexp(0.0, th), -np.logaddexp(0.0, -th)], axis=1)
        lb = lphi.copy()
        np.add.at(lb, q, m_up); np.add.at(lb, c, m_dn)
        cav_c, cav_p = lb[c] - m_dn, lb[q] - m_up
        new_up = np.stack([cav_c[:, 0], np.logaddexp(cav_c[:, 0], cav_c[:, 1])], axis=1)
        new_dn = np.stack([np.logaddexp(cav_p[:, 0], cav_p[:, 1]), cav_p[:, 1]], axis=1)
        new_up -= np.logaddexp(new_up[:, :1], new_up[:, 1:])
        new_dn -= np.logaddexp(new_dn[:, :1], new_dn[:, 1:])
        up = damping * np.exp(m_up) + (1 - damping) * np.exp(new_up)
        dn = damping * np.exp(m_dn) + (1 - damping) * np.exp(new_dn)
        delta = max(np.abs(up - np.exp(m_up)).max(), np.abs(dn - np.exp(m_dn)).max())
        m_up, m_dn = np.log(np.maximum(up, 1e-300)), np.log(np.maximum(dn, 1e-300))
        if delta < tol:
            break
    from wmc_inference import loopy_bp_theta
    r = loopy_bp_theta(n, edges, th, init=(m_up, m_dn), tol=1e-12, max_iter=5000)
    return th, float(np.abs(r["mu"] - mu_t).max()), it


def moment_match_bethe(n, edges, mu, gap=None):
    """Closed-form Bethe moment matching for implication theories. For an edge c => p the pairwise
    pseudo-marginal is fixed by the singletons (tau(1,1)=mu_c, tau(0,1)=mu_p-mu_c, tau(0,0)=1-mu_p),
    and the Bethe reparameterisation gives unary log-odds
        theta_i = logit(mu_i) + sum_{parents p} log((1-mu_i)/(mu_p-mu_i)) + sum_{children c} log((mu_i-mu_c)/mu_i)
    whose BP fixed point has beliefs exactly mu (exact on trees). Needs mu_c < mu_p strictly."""
    c, q = edges[:, 0], edges[:, 1]
    th = logit(mu).astype(float).copy()
    if gap is None:                     # pass gap = mu_p - mu_c explicitly when it is below float resolution
        gap = mu[q] - mu[c]
    if np.any(gap <= 0):
        raise ValueError("Bethe moment matching needs strictly ordered marginals (mu_child < mu_parent)")
    np.add.at(th, c, np.log1p(-mu[c]) - np.log(gap))
    np.add.at(th, q, np.log(gap) - np.log(mu[q]))
    return th


def moment_match_bp_logodds(n, edges, mu_t, damping=0.5, tol=1e-10, max_iter=50000):
    """BP-native moment matching with log-ratio messages: iterate theta = logit(mu_t) - incoming
    and one damped log-odds message update; at a fixed point the BP beliefs equal mu_t.
    Returns (theta, max belief error, iterations, converged)."""
    c, q = edges[:, 0], edges[:, 1]
    lt = logit(np.clip(mu_t, 1e-15, 1 - 1e-15))
    E = len(edges)
    r_up, r_dn = np.zeros(E), np.zeros(E)
    resid = np.inf
    for it in range(1, max_iter + 1):
        inc = np.zeros(n)
        np.add.at(inc, q, r_up); np.add.at(inc, c, r_dn)
        th = lt - inc
        b = lt                                      # beliefs are pinned to the target by construction
        new_up = np.logaddexp(0.0, b[c] - r_dn)
        new_dn = -np.logaddexp(0.0, -(b[q] - r_up))
        resid = max(np.abs(new_up - r_up).max(), np.abs(new_dn - r_dn).max())
        r_up = damping * r_up + (1 - damping) * new_up
        r_dn = damping * r_dn + (1 - damping) * new_dn
        if resid < tol:
            break
    inc = np.zeros(n); np.add.at(inc, q, r_up); np.add.at(inc, c, r_dn)
    th = lt - inc
    from wmc_inference import loopy_bp_logodds
    r = loopy_bp_logodds(n, edges, th, init=(r_up, r_dn), tol=1e-12)
    return th, float(np.abs(r["mu"] - mu_t).max()), it, bool(resid < tol)
