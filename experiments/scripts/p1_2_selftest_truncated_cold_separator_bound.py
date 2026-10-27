"""P1.2 theory self-test (plan section 8.4, Proposition C): truncated cold-module separator bound
F_k <= F <= F_k (1 + e^L L^{k+1}/(k+1)!) checked by enumeration on random modules."""
# Numerical check of the truncated cold-module separator bound.
# Module M (internal DAG, child->parent), boundary parents R. Valid y_M given b: on-set S is
# up-closed inside M and every R-parent of S is on. F(b)/prod(1-p_M) = sum_S prod_{m in S} lam_m [R-parents of S on].
# Claim: F_k <= F <= F_k * (1 + e^L L^{k+1}/(k+1)!)  with F_k the |S|<=k truncation, L = sum lam.
import itertools, math, numpy as np
rng = np.random.default_rng(0); worst = 0.0; n_checks = 0
for t in range(400):
    m = int(rng.integers(1, 9)); r = int(rng.integers(1, 6))
    inner = [(i, j) for i in range(m) for j in range(i) if rng.random() < 0.3]   # i -> j inside M
    rpar = {i: [q for q in range(r) if rng.random() < 0.35] for i in range(m)}
    p = rng.uniform(1e-4, rng.choice([0.01, 0.05, 0.2]), m); lam = p / (1 - p); L = lam.sum()
    sets = []
    for mask in range(1 << m):
        S = [i for i in range(m) if mask >> i & 1]
        if all((mask >> j & 1) for i, j in inner if mask >> i & 1):   # up-closed in M
            sets.append((S, set(q for i in S for q in rpar[i])))
    for b in itertools.product([0, 1], repeat=r):
        F = sum(np.prod(lam[S]) for S, need in sets if all(b[q] for q in need))
        for k in range(0, 4):
            Fk = sum(np.prod(lam[S]) for S, need in sets if len(S) <= k and all(b[q] for q in need))
            bound = math.exp(L) * L ** (k + 1) / math.factorial(k + 1)
            assert Fk <= F + 1e-15 and F <= Fk * (1 + bound) + 1e-15, (t, k, F, Fk, bound)
            if F > Fk: worst = max(worst, (F / Fk - 1) / bound)
            n_checks += 1
print(f"{n_checks} checks passed; worst (actual rel. error / bound) = {worst:.3f}")
