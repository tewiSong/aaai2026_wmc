"""P4.8 self-test: log-odds BP (wmc_inference.loopy_bp_logodds).
1. Codex counterexample C => A, theta_A = -40, theta_C = 40: exact marginals (0.5, 0.5);
2. random trees with strong weights theta ~ N(0, 15^2): equals exact circuit marginals;
3. on a loopy graph with moderate weights it reaches the same fixed point as loopy_bp."""
import tempfile
from pathlib import Path
import numpy as np
from scipy.special import expit
from learned_constrained import log_partition
from wmc_common import write_dimacs
from wmc_inference import Circuit, compile_with_d4, loopy_bp, loopy_bp_logodds, loopy_bp_theta

def circ(n, e):
    td = tempfile.mkdtemp(); write_dimacs(Path(td) / "t.cnf", n, e); compile_with_d4(Path(td) / "t.cnf", Path(td) / "t.nnf")
    return Circuit(Path(td) / "t.nnf", n)

e2 = np.array([[1, 0]], dtype=np.int32); th2 = np.array([-40.0, 40.0])
new = loopy_bp_logodds(2, e2, th2); old = loopy_bp_theta(2, e2, th2, tol=1e-10)
print("2-node: log-odds BP", new["mu"].round(9), "old BP", old["mu"].round(9))
assert np.abs(new["mu"] - 0.5).max() < 1e-9
rng = np.random.default_rng(0); worst = 0.0
for t in range(30):
    n = int(rng.integers(3, 40)); tree = np.array([(i, int(rng.integers(0, i))) for i in range(1, n)], dtype=np.int32)
    th = rng.normal(0, 15, n)
    worst = max(worst, np.abs(loopy_bp_logodds(n, tree, th, tol=1e-12)["mu"] - log_partition(circ(n, tree), th)[1]).max())
print(f"30 strong-weight trees: max |BP - exact| {worst:.1e}")
assert worst < 1e-8
n = 15; loopy = np.array(sorted({(i, int(rng.integers(0, i))) for i in range(1, n)} | {(i, j) for i in range(3, n) for j in range(1, i) if rng.random() < 0.2}), dtype=np.int32)
p = rng.uniform(0.1, 0.9, n)
d = np.abs(loopy_bp_logodds(n, loopy, np.log(p / (1 - p)), tol=1e-12)["mu"] - loopy_bp(n, loopy, p)["mu"]).max()
print(f"loopy moderate weights: |log-odds BP - prob BP| {d:.1e}")
assert d < 1e-7
print("P4.8 self-test passed")
