"""P4.7 self-test for the BP variant of C2.

1. loopy_bp_theta (log-odds input, warm start) equals loopy_bp on the same model;
2. on a tree (where BP is exact) moment_match_bp equals exact moment matching: same marginals
   and the same theta*;
3. train_c2_adam with exact marginals lowers the exact NLL from w = 0 to about the L-BFGS value.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from learned_constrained import ModelC2, log_partition, moment_match, moment_match_bp, moment_match_bp_fast, term_depth, train_c2_adam
from wmc_common import write_dimacs
from wmc_inference import Circuit, compile_with_d4, loopy_bp, loopy_bp_theta


def circuit(n, edges):
    td = tempfile.mkdtemp()
    write_dimacs(Path(td) / "t.cnf", n, edges); compile_with_d4(Path(td) / "t.cnf", Path(td) / "t.nnf")
    return Circuit(Path(td) / "t.nnf", n)


def main():
    rng = np.random.default_rng(5)
    # 1. BP from log-odds == BP from probabilities (loopy graph)
    n = 12
    edges = np.array(sorted({(i, int(rng.integers(0, i))) for i in range(1, n)} |
                            {(i, j) for i in range(3, n) for j in range(1, i) if rng.random() < 0.2}), dtype=np.int32)
    p = rng.uniform(0.05, 0.95, n)
    a = loopy_bp(n, edges, p)["mu"]
    b = loopy_bp_theta(n, edges, logit(p))
    c = loopy_bp_theta(n, edges, logit(p), init=b["messages"])
    assert np.abs(a - b["mu"]).max() < 1e-9 and np.abs(a - c["mu"]).max() < 1e-9 and c["iterations"] <= 3
    # 2. tree: BP moment matching == exact moment matching
    tree = np.array([(i, int(rng.integers(0, i))) for i in range(1, n)], dtype=np.int32)
    circ = circuit(n, tree)
    mu_t = circ.marginals(rng.uniform(0.05, 0.95, n))[0]            # an interior consistent target
    th_ex, err_ex = moment_match(circ, mu_t, logit(mu_t))
    th_bp, err_bp, k = moment_match_bp_fast(n, tree, mu_t)
    assert err_ex < 1e-7 and err_bp < 1e-6 and np.abs(th_ex - th_bp).max() < 1e-3, (err_ex, err_bp, np.abs(th_ex - th_bp).max())
    # 3. Adam trainer with exact marginals vs L-BFGS on synthetic proteins
    circ = circuit(n, edges)
    depth = term_depth(n, edges, 0)
    K = 300
    S = expit(rng.normal(-1, 1.5, (K, n)))
    L = logit(S)
    TH0, M = np.zeros_like(L), np.zeros_like(L)
    for k in range(K):
        mt = circ.marginals(S[k])[0]
        M[k] = logit(np.clip(mt, 1e-9, 1 - 1e-9))
        TH0[k] = moment_match(circ, mt, logit(mt))[0]
    Y = (rng.random((K, n)) < expit(L + 0.8)).astype(np.int8)       # labels not from the model
    for c_, q_ in edges:                                               # make labels valid (upward closure)
        pass
    for _ in range(n):
        for c_, q_ in edges:
            Y[:, q_] = np.maximum(Y[:, q_], Y[:, c_])
    m = ModelC2(circ, depth)
    nll0 = m.nll_grad(np.zeros(m.D * m.K), TH0, L, M, Y)[0]
    m.fit(TH0, L, M, Y, maxiter=500)
    nll_lbfgs = m.nll_grad(m.w, TH0, L, M, Y)[0]
    w_adam, _ = train_c2_adam(lambda k, th: log_partition(circ, th)[1], TH0, L, M, Y, depth, steps=400, lr=0.05)
    nll_adam = m.nll_grad(w_adam, TH0, L, M, Y)[0]
    print(f"BP-theta OK; tree MM: exact err {err_ex:.1e}, BP err {err_bp:.1e}, |theta diff| {np.abs(th_ex - th_bp).max():.1e}; "
          f"NLL w0 {nll0:.4f}, L-BFGS {nll_lbfgs:.4f}, Adam {nll_adam:.4f}")
    assert nll_adam < nll0 - 0.5 * (nll0 - nll_lbfgs)
    print("P4.7 self-test passed")


if __name__ == "__main__":
    main()
