"""P6.1 self-test of HotCoreHybrid (wmc_inference): (1) core = all terms reproduces exact
marginals; (2) on small random DAGs, hybrid marginals vs brute force for several core sizes, and
hybrid moment matching reproduces its own target; (3) one GO mf protein: hybrid at the exact theta*
vs the exact marginals (= target), for tau in {0.03, 0.01, 0.003}."""
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import logit

from consistency import isotonic_dag_hildreth
from wmc_common import DATA_DIR, load_theory, write_dimacs
from wmc_inference import Circuit, HotCoreHybrid, compile_with_d4, exact_marginals_brute
from scipy.special import expit

rng = np.random.default_rng(1)
td = Path(tempfile.mkdtemp())
worst = 0.0
low_gap = 0.0
mm_bad = mm_n = 0
for trial in range(30):
    n = 14
    edges = []
    for c in range(1, n):
        for p in rng.choice(c, size=min(c, rng.integers(1, 3)), replace=False):
            edges.append((c, int(p)))
    edges = np.array(edges)
    th = rng.normal(-1.0, 2.0, n); th[0] = 3.0
    mu_ex, _ = exact_marginals_brute(n, edges, expit(th))
    for k in (n, 8, 3, 1):
        core = np.zeros(n, bool); core[:k] = True             # parents have smaller index -> ancestor-closed
        hy = HotCoreHybrid(n, edges, core, td)
        r = hy.marginals(th, tol=1e-12, max_outer=5000)
        err = np.abs(r["mu"] - mu_ex).max()
        if k == n:
            assert err < 1e-9, err
        worst = max(worst, err) if k < n else worst
        mm = hy.moment_match(np.clip(mu_ex, 1e-9, 1 - 1e-9))
        r2 = hy.marginals(mm["theta"], init=mm["messages"], tol=1e-12, max_outer=5000)
        if k == n:
            assert np.abs(r2["mu"] - mu_ex).max() < 1e-6, (trial, np.abs(r2["mu"] - mu_ex).max())
        elif mm["converged"]:
            mm_bad += int(np.abs(r2["mu"] - mu_ex).max() > 1e-6); mm_n += 1
        r3 = hy.marginals(mm["theta"], init="low", tol=1e-12, max_outer=5000)
        low_gap = max(low_gap, np.abs(r3["mu"] - mu_ex).max())
print(f"(1)(2) passed; worst approximate-core error on random 14-node DAGs {worst:.3g}; max gap between the moment-matched fixed point and the low-start fixed point {low_gap:.3g}; moment-matched fixed point not reproduced by warm-started iteration (unstable) in {mm_bad}/{mm_n} converged cases")

ids, edges = load_theory("dgp2_go_mf"); n = len(ids)
z = np.load(DATA_DIR / "scores" / "deepgoplus2_go_mf.npz")
S, Y, root = z["cnn"].astype(float), z["labels"], int(z["root_index"])
keep = np.flatnonzero(np.delete(Y, root, 1).sum(1) > 0)
write_dimacs(td / "t.cnf", n, edges); compile_with_d4(td / "t.cnf", td / "t.nnf")
circ = Circuit(td / "t.nnf", n)
mu_unif, _ = circ.marginals(np.full(n, 0.5))
TH = np.load(DATA_DIR / "scores" / "theta_star_go_mf_cnn_eps0.001_v2.npz")["theta"]
for k in keep[:3]:
    s = np.clip(S[k], 1e-6, 1 - 1e-6)
    mt = 0.999 * isotonic_dag_hildreth(s[None], edges)[0][0] + 0.001 * mu_unif
    mu_ex = circ.marginals_theta(TH[k])[0]
    for tau in (0.03, 0.01, 0.003):
        hy = HotCoreHybrid(n, edges, mt >= tau, td)
        r = hy.marginals(TH[k], init="low", tol=1e-10)
        mm = hy.moment_match(mt)
        r_low = hy.marginals(mm["theta"], init="low", tol=1e-10)
        mu_back = circ.marginals_theta(mm["theta"])[0]
        print(f"protein {k} tau {tau}: core {len(hy.H)}, |hyb(theta*) - exact| max {np.abs(r['mu'] - mu_ex).max():.2e} "
              f"(iters {r['iterations']}, conv {r['converged']}); hybrid MM conv {mm['converged']} iters {mm['iterations']}, "
              f"|exact(theta_hyb) - target| max {np.abs(mu_back - mt).max():.2e}; |theta_hyb - theta*| median "
              f"{np.median(np.abs(mm['theta'] - TH[k])):.2e}; low-start at theta_hyb vs target {np.abs(r_low['mu'] - mt).max():.2e}", flush=True)
