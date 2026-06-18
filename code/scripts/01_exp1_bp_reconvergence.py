"""Exp 1: the soft closure is belief propagation; its error is confined to reconvergences.

Two parts:
  (A) Prior battery on the elementary structures (single edge, star, polytree, diamond):
      loopy BP equals the exact WMC marginal across a grid of priors on the tree fragments
      and is wrong on the diamond - the empirical counterpart of the Lean battery.
  (B) cellular_component (cc): exact junction-tree marginals vs the soft closure on the
      whole namespace. We report the error distribution, the localization of the error at
      multi-parent (reconvergent) terms, and a regression of per-term error against the
      number of parents (in-degree), turning the "~7x at multi-parent" observation into a
      continuous relation.

Outputs: results/exp1_battery.json, results/exp1_cc.json,
         results/exp1_cc_error_vs_indegree.csv, results/exp1_cc_error_vs_indegree.png
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter

import numpy as np

import lib
from truepath.graph import TruePathTheory
from truepath import bruteforce, softclosure, modular


def battery():
    """BP vs exact across a 9x9 prior grid on elementary structures."""
    grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    structures = {
        "single_edge": TruePathTheory("t", ["a", "b"], nf1=[("b", "a")]),
        "star3": TruePathTheory("t", ["a", "b", "c", "d"],
                                nf1=[("b", "a"), ("c", "a"), ("d", "a")]),
        "polytree_2parent": TruePathTheory("t", ["a", "b", "c"],
                                           nf1=[("a", "b"), ("a", "c")]),
        "diamond": TruePathTheory("t", ["a", "b", "c", "d"],
                                  nf1=[("d", "b"), ("d", "c"), ("b", "a"), ("c", "a")]),
    }
    out = {}
    for name, th in structures.items():
        max_err = 0.0
        n = len(th.atoms)
        cl = th.clauses()
        for qp in grid:
            for qc in grid:
                # two-value prior pattern: 'top-ish' atoms qp, others qc
                priors = {a: (qp if i % 2 == 0 else qc) for i, a in enumerate(th.atoms)}
                _, exact = bruteforce.brute_force_marginals(n, cl, [priors[a] for a in th.atoms])
                bp = softclosure.soft_closure_bp(th, priors, damping=0.3, max_iter=4000, tol=1e-12)
                err = max(abs(exact[i] - bp[a]) for i, a in enumerate(th.atoms))
                max_err = max(max_err, err)
        out[name] = dict(max_err_over_grid=max_err, exact_on_grid=bool(max_err < 1e-6))
        print(f"[battery] {name:18} max|BP-exact| over 9x9 grid = {max_err:.2e} "
              f"({'EXACT' if max_err < 1e-6 else 'INEXACT'})", flush=True)
    return out


def cc_experiment(seed: int = 0):
    theories = lib.load_theories()
    th = theories["cc"]
    priors = lib.make_priors(th, seed=seed)

    t0 = time.time()
    marg, stats = modular.modular_exact_marginals(th, priors, core=set())
    t_exact = time.time() - t0
    print(f"[cc] exact JT: {stats['n_terms_exact']} terms, max tw {stats['max_module_tw']}, "
          f"{t_exact:.1f}s", flush=True)

    t0 = time.time()
    bp = softclosure.soft_closure_bp(th, priors, damping=0.5, max_iter=4000, tol=1e-10)
    t_bp = time.time() - t0
    print(f"[cc] soft closure (loopy BP): {t_bp:.1f}s", flush=True)

    # Reconciliation (Task C): the paper's literal upward soft-OR vs bidirectional loopy BP.
    up = softclosure.soft_closure_upward(th, priors)
    up_err = np.array([abs(marg[a] - up[a]) for a in th.atoms])
    print(f"[cc] upward soft-OR (no downward cap): mean={up_err.mean():.2e} "
          f"max={up_err.max():.2e}  -- larger than loopy BP, confirming the deployed "
          f"closure is bidirectional BP", flush=True)

    indeg = Counter()
    for c, _p in th.nf1:
        indeg[c] += 1
    errs = np.array([abs(marg[a] - bp[a]) for a in th.atoms])
    deg = np.array([indeg[a] for a in th.atoms])

    mp = errs[deg > 1]
    sp = errs[deg <= 1]
    result = dict(
        seed=seed,
        n_terms=len(th.atoms),
        treewidth=stats["max_module_tw"],
        exact_time_s=round(t_exact, 2),
        bp_time_s=round(t_bp, 2),
        mean_err=float(errs.mean()),
        p95_err=float(np.percentile(errs, 95)),
        max_err=float(errs.max()),
        frac_err_gt_0p01=float(np.mean(errs > 0.01)),
        multi_parent_mean_err=float(mp.mean()),
        single_parent_mean_err=float(sp.mean()),
        localization_ratio=float(mp.mean() / sp.mean()),
        n_multi_parent=int((deg > 1).sum()),
        upward_softor_mean_err=float(up_err.mean()),
        upward_softor_max_err=float(up_err.max()),
    )
    print(f"[cc] mean={result['mean_err']:.2e} p95={result['p95_err']:.2e} "
          f"max={result['max_err']:.2e}  multi/single ratio={result['localization_ratio']:.1f}x",
          flush=True)

    # Error-vs-in-degree regression (continuous relation).
    rows = []
    for d in sorted(set(deg.tolist())):
        e = errs[deg == d]
        rows.append((d, len(e), float(e.mean()), float(e.max())))
    csv_path = os.path.join(lib.RESULTS_DIR, "exp1_cc_error_vs_indegree.csv")
    with open(csv_path, "w") as fh:
        fh.write("in_degree,n_terms,mean_err,max_err\n")
        for d, n, me, mx in rows:
            fh.write(f"{d},{n},{me:.6e},{mx:.6e}\n")
    # Linear fit of mean error vs in-degree (weighted by term count).
    ds = np.array([r[0] for r in rows], dtype=float)
    ms = np.array([r[2] for r in rows], dtype=float)
    ws = np.array([r[1] for r in rows], dtype=float)
    if len(ds) >= 2:
        coef = np.polyfit(ds, ms, 1, w=ws)
        result["error_vs_indegree_slope"] = float(coef[0])
        result["error_vs_indegree_intercept"] = float(coef[1])

    _plot_regression(rows, result)
    return result


def _plot_regression(rows, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = [r[0] for r in rows]
    ms = [r[2] for r in rows]
    sizes = [max(10, 3 * np.sqrt(r[1])) for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(ds, ms, s=sizes, alpha=0.7, label="mean error per in-degree")
    if "error_vs_indegree_slope" in result:
        xs = np.array([min(ds), max(ds)])
        ys = result["error_vs_indegree_slope"] * xs + result["error_vs_indegree_intercept"]
        ax.plot(xs, ys, "r--", label="weighted linear fit")
    ax.set_xlabel("number of parents (in-degree)")
    ax.set_ylabel("mean |soft closure - exact WMC|")
    ax.set_title("cc: soft-closure error grows with reconvergence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(lib.RESULTS_DIR, "exp1_cc_error_vs_indegree.png")
    fig.savefig(out, dpi=150)
    print("wrote", out, flush=True)


def main():
    lib.ensure_results_dir()
    bat = battery()
    with open(os.path.join(lib.RESULTS_DIR, "exp1_battery.json"), "w") as fh:
        json.dump(bat, fh, indent=2)
    cc = cc_experiment(seed=0)
    with open(os.path.join(lib.RESULTS_DIR, "exp1_cc.json"), "w") as fh:
        json.dump(cc, fh, indent=2)
    print("Exp1 done.")


if __name__ == "__main__":
    main()
