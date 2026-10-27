"""P1.2 self-test: the pruning certificate against brute force.

For random DAGs, score families and budgets: (i) the cold set is down-closed, (ii) the true
marginals lie in the certified intervals, (iii) the total-variation distance between the
true posterior and the pruned posterior (cold terms fixed to 0) is at most delta.
"""
from __future__ import annotations

import itertools
import tempfile
from pathlib import Path

import numpy as np

from wmc_inference import certified_marginals, cold_set_for_budget, make_scores


def joint(n, edges, p):
    ys = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int8)
    ok = np.ones(len(ys), dtype=bool)
    for c, q in edges:
        ok &= ~((ys[:, c] == 1) & (ys[:, q] == 0))
    w = np.where(ok, np.prod(np.where(ys == 1, p, 1 - p), axis=1), 0.0)
    return ys, w / w.sum()


def main():
    rng = np.random.default_rng(1)
    worst_slack, checks, nonempty = np.inf, 0, 0
    with tempfile.TemporaryDirectory() as td:
        for t in range(150):
            n = int(rng.integers(3, 14))
            edges = np.array([(i, j) for i in range(n) for j in range(i) if rng.random() < 0.35],
                             dtype=np.int32).reshape(-1, 2)
            if len(edges) == 0:
                edges = np.array([[1, 0]], dtype=np.int32)
            fam = ["sparse", "beta22", "uniform", "beta0505"][t % 4]
            p = make_scores(n, fam, rng)
            if fam == "sparse":
                p = np.where(rng.random(n) < 0.6, rng.uniform(1e-6, 1e-2, n), p)
            ys, true = joint(n, edges, p)
            mu = (true[:, None] * ys).sum(0)
            for target in (1e-3, 1e-2, 0.1, 0.5):
                in_d, delta, _ = cold_set_for_budget(n, edges, p, target)
                # down-closed: every child of a cold term is cold
                for c, q in edges:
                    assert not (in_d[q] and not in_d[c])
                res = certified_marginals(n, edges, p, target, Path(td))
                assert np.all(mu >= res["lo"] - 1e-12) and np.all(mu <= res["hi"] + 1e-12)
                pruned = np.where((ys[:, in_d] == 0).all(1), true, 0.0)
                pruned /= pruned.sum()
                tv = 0.5 * np.abs(true - pruned).sum()
                assert tv <= delta + 1e-12, (tv, delta)
                worst_slack = min(worst_slack, delta - tv)
                checks += 1
                nonempty += bool(in_d.any())
    print(f"{checks} checks ({nonempty} with a non-empty cold set): intervals contain the true "
          f"marginals and TV <= delta; min slack {worst_slack:.2e}")
    print("P1.2 certified-pruning self-test passed")


if __name__ == "__main__":
    main()
