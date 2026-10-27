"""P4.1 self-test: DAG isotonic regression (consistency.isotonic_dag).

1. chains: equals pool-adjacent-violators;
2. random DAGs: equals a generic constrained solver (SLSQP) on the same QP;
3. Bernoulli-KL projection min sum KL(Ber(p_i) || Ber(mu_i)) s.t. mu_c <= mu_p gives the same
   solution as the squared-error one (isotonic regression is simultaneously optimal for
   Bregman losses of this form) - checked numerically, so the KL variant of Obozinski et al.
   in this direction needs no separate implementation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from consistency import isotonic_dag, isotonic_dag_hildreth, pav_chain


def main():
    rng = np.random.default_rng(0)
    worst = dict(pav=0.0, slsqp=0.0, kl=0.0, hildreth=0.0)
    for t in range(40):                      # chains: node i is child of node i-1
        n = int(rng.integers(2, 30))
        p = rng.uniform(0, 1, n)
        edges = np.array([(i, i - 1) for i in range(1, n)], dtype=np.int32)
        mu, info = isotonic_dag(p[None], edges)
        worst["pav"] = max(worst["pav"], np.abs(mu[0] - pav_chain(p)).max())
    for t in range(40):
        n = int(rng.integers(3, 12))
        edges = np.array([(i, j) for i in range(n) for j in range(i) if rng.random() < 0.35],
                         dtype=np.int32).reshape(-1, 2)
        if len(edges) == 0:
            continue
        p = np.clip(rng.beta(0.6, 0.6, n), 1e-4, 1 - 1e-4)
        mu = isotonic_dag(p[None], edges)[0][0]
        cons = [{"type": "ineq", "fun": (lambda x, c=c, q=q: x[q] - x[c])} for c, q in edges]
        ref = minimize(lambda x: 0.5 * np.sum((x - p) ** 2), p, jac=lambda x: x - p,
                       constraints=cons, method="SLSQP", options=dict(ftol=1e-14, maxiter=1000)).x
        worst["slsqp"] = max(worst["slsqp"], np.abs(mu - ref).max())
        worst["hildreth"] = max(worst["hildreth"], np.abs(isotonic_dag_hildreth(p[None], edges, tol=1e-12)[0][0] - ref).max())
        kl = lambda x: np.sum(p * np.log(p / x) + (1 - p) * np.log((1 - p) / (1 - x)))
        klg = lambda x: -p / x + (1 - p) / (1 - x)
        ref_kl = minimize(kl, mu, jac=klg, constraints=cons, bounds=[(1e-6, 1 - 1e-6)] * n,
                          method="SLSQP", options=dict(ftol=1e-15, maxiter=2000)).x
        worst["kl"] = max(worst["kl"], np.abs(mu - ref_kl).max())
    print("max abs differences:", {k: f"{v:.2e}" for k, v in worst.items()})
    assert worst["pav"] < 1e-8 and worst["slsqp"] < 1e-6 and worst["kl"] < 1e-5
    print("P4.1 isotonic self-test passed")


if __name__ == "__main__":
    main()
