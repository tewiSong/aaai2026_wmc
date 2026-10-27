"""P4.5 self-test: learned_constrained.log_partition (stable version) against brute force for
extreme log-odds (theta ~ N(0, 25^2)) on 200 random DAGs. Regression test for the clipping
bug found by the Codex P4.4-P4.6 audit (single edge, theta=(-40, 40): true 0.693, old 13.06)."""
import itertools, tempfile
from pathlib import Path
import numpy as np
from learned_constrained import log_partition
from wmc_common import write_dimacs
from wmc_inference import Circuit, compile_with_d4

rng = np.random.default_rng(0); worst = worstmu = 0.0
for t in range(200):
    n = int(rng.integers(2, 10))
    e = np.array([(i, j) for i in range(n) for j in range(i) if rng.random() < 0.4], dtype=np.int32).reshape(-1, 2)
    if len(e) == 0:
        e = np.array([[1, 0]], dtype=np.int32)
    with tempfile.TemporaryDirectory() as td:
        write_dimacs(Path(td) / "t.cnf", n, e); compile_with_d4(Path(td) / "t.cnf", Path(td) / "t.nnf")
        c = Circuit(Path(td) / "t.nnf", n)
    th = rng.normal(0, 25, n)
    if t == 0:
        n, e, th = 2, np.array([[1, 0]], dtype=np.int32), np.array([-40.0, 40.0])
        with tempfile.TemporaryDirectory() as td:
            write_dimacs(Path(td) / "t.cnf", n, e); compile_with_d4(Path(td) / "t.cnf", Path(td) / "t.nnf")
            c = Circuit(Path(td) / "t.nnf", n)
    ys = np.array([y for y in itertools.product([0, 1], repeat=n) if all(not (y[a] == 1 and y[b] == 0) for a, b in e)])
    sc = ys @ th; m = sc.max(); true = m + np.log(np.exp(sc - m).sum())
    mu_true = (np.exp(sc - true)[:, None] * ys).sum(0)
    lz, mu = log_partition(c, th)
    worst = max(worst, abs(lz - true)); worstmu = max(worstmu, np.abs(mu - mu_true).max())
print(f"max |logZ err| {worst:.2e}, max |marginal err| {worstmu:.2e}")
assert worst < 1e-9 and worstmu < 1e-9
print("stable log-partition self-test passed")
