"""P6.0  Regime diagnostic for exact inference at GO bp scale (plan item (d)).

d4 cannot compile the whole GO bp vocabulary theory (P1.4a: segfault after 29.6 h with ~4.3e9
unit literals), so C2 needs another inference route there. Before choosing one, measure per
protein how the isotonic target mu* splits into a hot core and a cold remainder:
  H(tau) = {i : mu*_i >= tau}   (ancestor-closed, since mu* is isotonic)
  Lambda(tau) = sum_{i not in H} mu*_i / (1 - mu*_i)   (cold odds mass; the truncation bounds
               of plan Prop C and the pruning certificate both scale with it)
and how long d4 takes on the hot-core theory H(tau) for a sample of proteins.
GO cc / GO mf (where exact inference works) are measured the same way for reference.
Output: results/p6_0_hot_core_regime_<ns>_<input>.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from consistency import isotonic_dag_hildreth
from wmc_common import DATA_DIR, RESULTS_DIR, load_theory, write_dimacs
from wmc_inference import compile_with_d4

TAUS = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
COMPILE_TAUS = [0.03, 0.01, 0.003, 0.001]


def sub_theory(core, edges, n):
    idx = -np.ones(n, dtype=np.int64)
    idx[core] = np.arange(len(core))
    keep = (idx[edges[:, 0]] >= 0) & (idx[edges[:, 1]] >= 0)
    return idx[edges[keep]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True)
    ap.add_argument("--input", choices=["cnn", "fused"], required=True)
    ap.add_argument("--proteins", type=int, default=400)
    ap.add_argument("--compile-sample", type=int, default=20)
    ap.add_argument("--d4-timeout", type=float, default=300.0)
    args = ap.parse_args()

    ids, edges = load_theory(f"dgp2_{args.ns}")
    n = len(ids)
    z = np.load(DATA_DIR / "scores" / f"deepgoplus2_{args.ns}.npz")
    S, Y, root = z[args.input].astype(float), z["labels"], int(z["root_index"])
    cols = np.array([j for j in range(n) if j != root])
    keep = np.flatnonzero(Y[:, cols].sum(1) > 0)
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(keep, min(args.proteins, len(keep)), replace=False))
    S = np.clip(S[pick], 1e-6, 1 - 1e-6)
    t0 = time.time()
    MU, iso_info = isotonic_dag_hildreth(S, edges)
    MU = np.clip(MU, 1e-6, 1 - 1e-6)
    print(f"isotonic {time.time() - t0:.1f}s {iso_info}", flush=True)

    odds = MU / (1 - MU)
    per_tau = {}
    for tau in TAUS:
        hot = MU >= tau
        size = hot.sum(1)
        lam = np.where(hot, 0.0, odds).sum(1)
        per_tau[str(tau)] = dict(core_size=np.quantile(size, [0.1, 0.5, 0.9, 1.0]).tolist(),
                                 cold_odds_mass=np.quantile(lam, [0.1, 0.5, 0.9, 1.0]).tolist())
        print(f"tau {tau:<6} core q10/50/90/max {per_tau[str(tau)]['core_size']}  "
              f"Lambda {np.round(per_tau[str(tau)]['cold_odds_mass'], 3).tolist()}", flush=True)
    total_odds = odds.sum(1)

    compile_stats = {}
    td = Path(tempfile.mkdtemp())
    sample = rng.choice(len(pick), min(args.compile_sample, len(pick)), replace=False)
    for tau in COMPILE_TAUS:
        recs = []
        for k in sample:
            core = np.flatnonzero(MU[k] >= tau)
            sub = sub_theory(core, edges, n)
            write_dimacs(td / "c.cnf", len(core), sub)
            try:
                r = compile_with_d4(td / "c.cnf", td / "c.nnf", timeout_s=args.d4_timeout)
                recs.append(dict(core=int(len(core)), seconds=r["seconds"],
                                 nnf_mb=round((td / "c.nnf").stat().st_size / 2 ** 20, 2)))
            except Exception as e:  # timeout or crash
                recs.append(dict(core=int(len(core)), seconds=None, error=type(e).__name__))
        ok = [r["seconds"] for r in recs if r["seconds"] is not None]
        compile_stats[str(tau)] = dict(records=recs, solved=len(ok), of=len(recs),
                                       seconds_median=float(np.median(ok)) if ok else None)
        print(f"compile tau {tau}: solved {len(ok)}/{len(recs)}, median {compile_stats[str(tau)]['seconds_median']} s, "
              f"core sizes {[r['core'] for r in recs]}", flush=True)
        if not ok:
            break

    out = dict(theory=f"dgp2_{args.ns}", input=args.input, terms=n, proteins=int(len(pick)), isotonic=iso_info,
               total_odds_mass=np.quantile(total_odds, [0.1, 0.5, 0.9, 1.0]).tolist(), per_tau=per_tau,
               compile=compile_stats)
    json.dump(out, open(RESULTS_DIR / f"p6_0_hot_core_regime_{args.ns}_{args.input}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
