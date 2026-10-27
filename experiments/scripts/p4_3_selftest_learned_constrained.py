"""P4.3 self-test for the learned constrained model C (scripts/learned_constrained.py).

On a small rooted DAG (exact enumeration available):
1. with a = 1, b = 0 model C reproduces plain exact conditioning of the scores;
2. the analytic gradient of the mean negative log-likelihood matches finite differences;
3. parameters are recovered when labels are sampled exactly from a model C with known a, b.
"""
from __future__ import annotations

import itertools
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from learned_constrained import ModelC, term_depth
from wmc_common import write_dimacs
from wmc_inference import Circuit, compile_with_d4, exact_marginals_brute


def main():
    rng = np.random.default_rng(3)
    n = 10
    edges = [(i, int(rng.integers(0, i))) for i in range(1, n)]               # tree to root 0
    edges += [(i, j) for i in range(3, n) for j in range(1, i) if rng.random() < 0.15 and (i, j) not in edges]
    edges = np.array(sorted(set(edges)), dtype=np.int32)
    depth = term_depth(n, edges, 0)
    with tempfile.TemporaryDirectory() as td:
        write_dimacs(Path(td) / "t.cnf", n, edges)
        compile_with_d4(Path(td) / "t.cnf", Path(td) / "t.nnf")
        circ = Circuit(Path(td) / "t.nnf", n)
    model = ModelC(circ, depth, ridge=0.0)
    D = model.D

    # 1. identity parameters = plain conditioning
    s = np.clip(rng.uniform(0.02, 0.98, n), 1e-6, 1 - 1e-6)
    model.params = np.r_[np.ones(D), np.zeros(D)]
    mu_c = model.predict(logit(s)[None])[0]
    mu_ref, _ = exact_marginals_brute(n, edges, s)
    assert np.abs(mu_c - mu_ref).max() < 1e-10

    # labels sampled exactly from a model C with known parameters
    true = np.r_[rng.uniform(0.6, 1.6, D), rng.uniform(-1.5, 1.0, D)]
    ys = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int8)
    ok = np.ones(len(ys), bool)
    for c, q in edges:
        ok &= ~((ys[:, c] == 1) & (ys[:, q] == 0))
    ys = ys[ok]
    K = 3000
    L = logit(np.clip(rng.uniform(0.02, 0.98, (K, n)), 1e-6, 1 - 1e-6))
    Y = np.zeros((K, n), dtype=np.int8)
    for k in range(K):
        th = model.theta(true, L[k])
        w = np.exp(ys @ th)
        Y[k] = ys[rng.choice(len(ys), p=w / w.sum())]

    # 2. gradient check
    x = true + rng.normal(0, 0.3, 2 * D)
    f0, g = model.nll_grad(x, L[:200], Y[:200])
    num = np.zeros_like(x)
    for j in range(len(x)):
        e = np.zeros_like(x); e[j] = 1e-6
        num[j] = (model.nll_grad(x + e, L[:200], Y[:200])[0] - model.nll_grad(x - e, L[:200], Y[:200])[0]) / 2e-6
    assert np.abs(num - g).max() < 1e-5, (num, g)

    # 3. parameter recovery
    model.fit(L, Y)
    err = np.abs(model.params - true).max()
    print(f"identity==conditioning OK; gradient max diff {np.abs(num - g).max():.2e}; "
          f"recovered params max abs error {err:.3f} (true range a in [0.6,1.6], b in [-1.5,1])")
    assert err < 0.25
    print("P4.3 self-test passed")


if __name__ == "__main__":
    main()
