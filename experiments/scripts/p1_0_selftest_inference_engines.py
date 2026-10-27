"""P1.0 self-test: every inference engine against independent references.

1. Diamond (paper's motivating example): exact P(y_a = 1) = 5/6 for brute force, d4
   circuit and dense VE.
2. Random small DAGs: d4-circuit marginals and VE log Z equal brute force to 1e-9;
   loopy BP is exact on forests; Gibbs agrees with brute force within Monte Carlo error.
3. GO cc at full scale: d4-circuit marginals equal dense VE marginals on sampled terms
   (two independent exact methods), and log Z agrees.
Run by Claude on the login node before any P1.0 batch job.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from wmc_common import ORDER_DIR, load_theory, write_dimacs
from wmc_inference import (Circuit, compile_with_d4, exact_marginals_brute, gibbs_marginals,
                           loopy_bp, make_scores, ve_log_z, violations)


def circuit_for(tmp: Path, n: int, edges: np.ndarray, tag: str) -> Circuit:
    cnf, nnf = tmp / f"{tag}.cnf", tmp / f"{tag}.nnf"
    write_dimacs(cnf, n, edges)
    compile_with_d4(cnf, nnf)
    return Circuit(nnf, n)


def ve_marginal(n, edges, p, order, i):
    lz = ve_log_z(n, edges, p, order)
    return np.exp(ve_log_z(n, edges, p, order, clamp={i: 1}) - lz), lz


def main() -> None:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # 1. diamond: d => b, d => c, b => a, c => a  (a=0, b=1, c=2, d=3)
        dia = np.array([[3, 1], [3, 2], [1, 0], [2, 0]], dtype=np.int32)
        half = np.full(4, 0.5)
        mu_b, _ = exact_marginals_brute(4, dia, half)
        mu_c, _ = circuit_for(tmp, 4, dia, "diamond").marginals(half)
        mu_v, _ = ve_marginal(4, dia, half, [3, 2, 1, 0], 0)
        for name, val in [("brute", mu_b[0]), ("d4", mu_c[0]), ("VE", mu_v)]:
            assert abs(val - 5 / 6) < 1e-12, (name, val)
        print("diamond: P(a)=5/6 for brute force, d4 circuit, VE")

        # 2. random small DAGs
        worst = dict(circuit=0.0, ve=0.0, bp_forest=0.0, gibbs=0.0)
        families = ["uniform", "beta22", "beta0505", "sparse"]
        for t in range(120):
            n = int(rng.integers(2, 15))
            dens = rng.uniform(0.05, 0.5)
            edges = np.array([(i, j) for i in range(n) for j in range(i) if rng.random() < dens],
                             dtype=np.int32).reshape(-1, 2)
            if len(edges) == 0:
                edges = np.array([[1, 0]], dtype=np.int32)
            circ = circuit_for(tmp, n, edges, f"r{t}")
            for fam in families:
                p = make_scores(n, fam, rng)
                ref, z = exact_marginals_brute(n, edges, p)
                mu, logz = circ.marginals(p)
                worst["circuit"] = max(worst["circuit"], np.abs(mu - ref).max(), abs(logz - np.log(z)))
                worst["ve"] = max(worst["ve"], abs(ve_log_z(n, edges, p, list(range(n))) - np.log(z)))
                assert violations(edges, mu) == 0
        # forests: one parent each, BP must be exact
        for t in range(60):
            n = int(rng.integers(2, 15))
            edges = np.array([(i, int(rng.integers(0, i))) for i in range(1, n)], dtype=np.int32)
            p = make_scores(n, families[t % 4], rng)
            ref, _ = exact_marginals_brute(n, edges, p)
            worst["bp_forest"] = max(worst["bp_forest"], np.abs(loopy_bp(n, edges, p)["mu"] - ref).max())
        # Gibbs on a few loopy graphs
        for t in range(5):
            n = 12
            edges = np.array([(i, j) for i in range(n) for j in range(i) if rng.random() < 0.3],
                             dtype=np.int32).reshape(-1, 2)
            p = make_scores(n, "beta22", rng)
            ref, _ = exact_marginals_brute(n, edges, p)
            g = gibbs_marginals(n, edges, p, rng, chains=32, burn_in=200, sweeps=3000)
            worst["gibbs"] = max(worst["gibbs"], np.abs(g["mu"] - ref).max())
        print("random DAGs, worst abs error:", {k: f"{v:.2e}" for k, v in worst.items()})
        assert worst["circuit"] < 1e-9 and worst["ve"] < 1e-9 and worst["bp_forest"] < 1e-8
        assert worst["gibbs"] < 1e-2

        # 3. GO cc: two independent exact methods at full scale
        ids, edges = load_theory("go_cc")
        n = len(ids)
        circ = circuit_for(tmp, n, edges, "go_cc")
        with np.load(ORDER_DIR / "go_cc_min-fill.npz") as z:
            order = z["order"].tolist()
        for fam in ["uniform", "sparse"]:
            p = make_scores(n, fam, rng)
            mu, logz = circ.marginals(p)
            sample = rng.choice(n, 25, replace=False)
            diffs, lz_ve = [], None
            for i in sample:
                m, lz_ve = ve_marginal(n, edges, p, order, int(i))
                diffs.append(abs(m - mu[i]))
            print(f"GO cc {fam}: circuit nodes={circ.n_nodes} edges={circ.n_edges}; "
                  f"|logZ diff|={abs(logz - lz_ve):.2e}; max marginal diff on 25 terms={max(diffs):.2e}")
            assert abs(logz - lz_ve) < 1e-8 and max(diffs) < 1e-9
            assert violations(edges, mu) == 0
    print("P1.0 self-test passed")


if __name__ == "__main__":
    main()
