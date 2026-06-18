"""Cost-II in the disjunctive (SROIQ) regime: a controlled multimodal boundary joint.

Two entities, each typed by an exactly-one-of-three axiom (covering + pairwise
disjointness), coupled by cross-entity disjointness (a_i and b_i cannot both hold). The
exact joint over the w=6 atoms is multimodal; the independence assumption (best product)
cannot represent it, and a rank-r tensor-train separator message is exact only at the
joint's full tensor-train rank. Reproduces Table~\ref{tab:disj}.
"""

from __future__ import annotations

import json
import os
from itertools import product

import numpy as np

import lib
from truepath import rank_dial as rd


def main():
    lib.ensure_results_dir()
    w = 6
    pri = [0.5] * w

    def exactly_one(x):
        return sum(x) == 1

    T = np.zeros((2,) * w)
    modes = 0
    for asg in product((0, 1), repeat=w):
        a, b = asg[:3], asg[3:]
        if not (exactly_one(a) and exactly_one(b)):
            continue
        if any(a[i] and b[i] for i in range(3)):  # cross-entity disjointness
            continue
        wt = 1.0
        for i in range(w):
            wt *= pri[i] if asg[i] else 1 - pri[i]
        T[asg] = wt
        modes += 1
    T /= T.sum()

    def composed_query(dist):  # P(a1 or b2)
        return float(sum(dist[asg] for asg in product((0, 1), repeat=w)
                         if asg[0] == 1 or asg[4] == 1))

    exact_q = composed_query(T)
    p1 = rd.product_marginal(T)
    exact_rank = rd.tt_max_rank(rd.tt_svd(T, max_rank=None))

    rows = []
    kl, tv = rd.kl_tv(T, p1)
    rows.append(dict(model="IA", kl=kl, tv=tv,
                     wmc_relerr=abs(exact_q - composed_query(p1)) / exact_q))
    for r in [2, 3]:
        cores = rd.tt_svd(T, max_rank=r)
        pr = np.clip(rd.tt_reconstruct(cores), 0, None)
        pr /= pr.sum()
        kl, tv = rd.kl_tv(T, pr)
        rows.append(dict(model=f"rank{r}", kl=kl, tv=tv,
                         wmc_relerr=abs(exact_q - composed_query(pr)) / exact_q))

    out = dict(modes=modes, exact_rank=exact_rank, rows=rows)
    print(f"[disjunctive] modes={modes} exact_rank={exact_rank}")
    for row in rows:
        print(f"  {row['model']:6}: KL={row['kl']:.3f} TV={row['tv']:.3f} "
              f"query_relerr={row['wmc_relerr']:.3e}")
    path = os.path.join(lib.RESULTS_DIR, "exp_disjunctive.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
