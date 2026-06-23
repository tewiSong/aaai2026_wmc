"""Exp 2: treewidth, modular exact WMC, the core-cut trade-off curve, and the rank-r dial.

Parts (selectable with --part):
  treewidth : min-fill width of each namespace's true-path constraint graph.
  corecut   : greedily remove highest-degree terms in 1% batches, recompute the treewidth
              of the disconnected non-core remainder, and record that residual diagnostic
              curve. This is not the separator-aware modular result.
  modular   : select the reconvergence core that drops the remainder to tw <= target,
              including each module's adjacent core boundary, run exact junction-tree WMC
              with weighted separator messages, time it, and validate a sample of augmented
              modules against brute force.
  rankdial  : extract real cc reconvergence cores (upper cones), compute the exact
              constraint-induced boundary joint, the rank-1 (independent / soft closure)
              gap, and the rank-r tensor-train dial to exact; report a single detailed
              core table plus statistics over the 40 highest-degree cores.

Outputs go to results/exp2_*.json / .csv.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import networkx as nx
import numpy as np

import lib
from truepath.graph import TruePathTheory
from truepath import treewidth as tw_mod
from truepath import modular, rank_dial, bruteforce, junction_tree, softclosure


# ----------------------------------------------------------------------------- treewidth
def run_treewidth(theories, namespaces):
    out = {}
    for ns in namespaces:
        th = theories[ns]
        g = th.constraint_graph()
        t0 = time.time()
        tw = tw_mod.treewidth_min_fill(g)
        dt = time.time() - t0
        out[ns] = dict(n_terms=len(th.atoms), treewidth=tw, time_s=round(dt, 1))
        print(f"[treewidth] {ns}: terms={len(th.atoms)} tw={tw} [{dt:.0f}s]", flush=True)
    _dump("exp2_treewidth.json", out)
    return out


# ------------------------------------------------------------------------------- corecut
def run_corecut(theories, namespaces, target_tw=14):
    out = {}
    for ns in namespaces:
        th = theories[ns]
        print(f"[corecut] {ns} starting (target tw <= {target_tw})", flush=True)
        records = modular.core_cut_curve(th, target_tw=target_tw, batch_frac=0.01, max_frac=0.30)
        out[ns] = records
        # also dump a CSV per namespace
        csv = os.path.join(lib.RESULTS_DIR, f"exp2_corecut_{ns}.csv")
        with open(csv, "w") as fh:
            fh.write("cut_count,cut_frac,max_tw,exact_frac,n_modules,tw_time_s\n")
            for r in records:
                fh.write(f"{r['cut_count']},{r['cut_frac']:.4f},{r['max_tw']},"
                         f"{r['exact_frac']:.4f},{r['n_modules']},{r['tw_time_s']:.2f}\n")
        print(f"[corecut] {ns} wrote {csv}", flush=True)
    _dump("exp2_corecut.json", out)
    return out


# ------------------------------------------------------------------------------- modular
def run_modular(theories, namespaces, target_tw=14, seed=0, validate_modules=20):
    out = {}
    for ns in namespaces:
        th = theories[ns]
        priors = lib.make_priors(th, seed=seed)
        print(f"[modular] {ns} selecting boundary-aware core (target tw <= {target_tw})", flush=True)
        t_select = time.time()
        core = modular.select_core_boundary_aware(th, target_tw=target_tw, max_frac=0.30)
        select_dt = time.time() - t_select
        if core:
            print(f"[modular] {ns} computing rank-1 separator message by soft-closure BP",
                  flush=True)
            t_bp = time.time()
            boundary_marg = softclosure.soft_closure_bp(
                th, priors, max_iter=20000, tol=1e-9, damping=0.5)
            bp_dt = time.time() - t_bp
        else:
            boundary_marg = {}
            bp_dt = 0.0
        t0 = time.time()
        marg, stats = modular.modular_marginals_with_boundary(
            th, priors, core=core, boundary_marg=boundary_marg, tw_cap=target_tw)
        dt = time.time() - t0
        stats = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}
        stats["core_selection_time_s"] = round(select_dt, 2)
        stats["boundary_bp_time_s"] = round(bp_dt, 2)
        stats["module_jt_time_s"] = round(dt, 2)
        stats["wall_time_s"] = round(select_dt + bp_dt + dt, 2)
        stats["boundary_message"] = "rank1_soft_closure_bp" if core else "none"
        # Validate a sample of small augmented modules against brute force.
        val_err, validation_checked = _validate_modules(
            th, priors, core, boundary_marg, n_modules=validate_modules,
            require_any=bool(core))
        stats["validation_max_abs_err"] = val_err
        stats["validation_checked"] = validation_checked
        out[ns] = stats
        val_msg = "not-applicable" if val_err is None else f"{val_err:.1e}"
        print(f"[modular] {ns}: core={stats['core_size']} ({100*stats['core_frac']:.1f}%) "
              f"modules={stats['n_modules']} max_tw={stats['max_module_tw']} "
              f"exact={100*stats['exact_frac']:.1f}% total={stats['wall_time_s']}s "
              f"jt={stats['module_jt_time_s']}s val_err={val_msg}", flush=True)
    _dump("exp2_modular.json", out)
    return out


def _validate_modules(th, priors, core, boundary_marg, n_modules=20, max_vars=24,
                      require_any=False):
    """Brute-force check exact JT marginals on small augmented modules."""
    g0 = th.constraint_graph()
    sub = g0.subgraph([v for v in th.atoms if v not in core])
    gidx = {a: i + 1 for i, a in enumerate(th.atoms)}
    clauses = th.clauses()
    max_err = 0.0
    checked = 0
    comps = sorted(nx.connected_components(sub), key=len)
    for comp in comps:
        if checked >= n_modules:
            break
        comp = set(comp)
        aug = modular.augmented_module_graph(th, core, comp, constraint_graph=g0)
        aug_nodes = sorted(aug.nodes())
        if len(aug_nodes) > max_vars:
            continue
        local_idx = {v: i for i, v in enumerate(aug_nodes)}
        local_clauses = []
        nset = {gidx[v] for v in aug_nodes}
        comp_gidx = {gidx[v] for v in comp}
        inv = {gidx[v]: v for v in aug_nodes}
        mod_clauses = []
        for cl in clauses:
            if any(abs(l) in comp_gidx for l in cl) and all(abs(l) in nset for l in cl):
                mod_clauses.append(cl)
                local_clauses.append(tuple(
                    (1 if l > 0 else -1) * (local_idx[inv[abs(l)]] + 1) for l in cl))
        pri = [boundary_marg[v] if v in core else priors[v] for v in aug_nodes]
        _, bf = bruteforce.brute_force_marginals(len(aug_nodes), local_clauses, pri)
        loc_priors = {v: (boundary_marg[v] if v in core else priors[v]) for v in aug_nodes}
        jt = junction_tree.JunctionTree(aug, gidx)
        jm = jt.calibrate_marginals(mod_clauses, loc_priors)
        for v in sorted(comp):
            max_err = max(max_err, abs(bf[local_idx[v]] - jm[v]))
        checked += 1
    if checked == 0 and sub.number_of_nodes() > 0:
        if not require_any:
            return None, 0
        raise RuntimeError(
            f"no augmented module with <= {max_vars} variables was available for validation"
        )
    return float(max_err), checked


# ------------------------------------------------------------------------------ rank dial
def extract_upper_cone(th: TruePathTheory, root, max_size=13) -> TruePathTheory:
    """Sub-theory induced by `root` and its ancestors (up the implication DAG), capped."""
    parents = {}
    for c, p in th.nf1:
        parents.setdefault(c, []).append(p)
    nodes = [root]
    seen = {root}
    i = 0
    while i < len(nodes) and len(nodes) < max_size:
        cur = nodes[i]
        i += 1
        for p in parents.get(cur, []):
            if p not in seen and len(nodes) < max_size:
                seen.add(p)
                nodes.append(p)
    nodes = sorted(seen)
    nset = set(nodes)
    sub_nf1 = [(c, p) for c, p in th.nf1 if c in nset and p in nset]
    return TruePathTheory("core", nodes, nf1=sub_nf1)


def run_rankdial(theories, seed=0, n_cores=40, max_size=13):
    th = theories["cc"]
    priors_full = lib.make_priors(th, seed=seed)
    indeg = Counter()
    for c, _p in th.nf1:
        indeg[c] += 1
    # rank multi-parent terms by in-degree as core seeds
    seeds = [a for a in sorted(th.atoms, key=lambda x: indeg[x], reverse=True) if indeg[a] > 1]

    detailed = None
    stats = []
    used = 0
    for s in seeds:
        if used >= n_cores:
            break
        core_th = extract_upper_cone(th, s, max_size=max_size)
        if len(core_th.atoms) < 4 or len(core_th.atoms) > 22:
            continue
        # multi-parent count inside the core
        cind = Counter()
        for c, p in core_th.nf1:
            cind[c] += 1
        n_mp = sum(1 for a in core_th.atoms if cind[a] > 1)
        if n_mp == 0:
            continue
        priors = {a: priors_full[a] for a in core_th.atoms}
        res = rank_dial.rank_dial(core_th, priors, ranks=[1, 2, 4, 8, 10, 12, 16])
        res["seed_term"] = s
        res["n_multi_parent"] = n_mp
        stats.append(res)
        used += 1
        if detailed is None and len(core_th.atoms) >= 10:
            detailed = res
            print(f"[rankdial] detailed core seed={s}: w={res['w']} models={res['n_models']} "
                  f"exact_rank={res['exact_rank']} KL_rank1={res['kl_rank1']:.2f} "
                  f"TV_rank1={res['tv_rank1']:.2f}", flush=True)
            for row in res["rows"]:
                proj = " projected" if row.get("projected") else ""
                print(f"    rank {row['rank']:2d}: KL={row['kl']:.3f} TV={row['tv']:.3f} "
                      f"gap_closed={row['gap_closed']:.0%}{proj}", flush=True)

    if detailed is None and stats:
        detailed = max(stats, key=lambda r: r["w"])
    if not stats:
        raise RuntimeError("rank dial did not find any valid reconvergence core")

    kl1 = np.array([r["kl_rank1"] for r in stats])
    exact_ranks = np.array([r["exact_rank"] for r in stats])
    dense = np.array([2 ** r["w"] for r in stats])
    summary = dict(
        n_cores=len(stats),
        kl_rank1_median=float(np.median(kl1)),
        kl_rank1_range=[float(kl1.min()), float(kl1.max())],
        exact_rank_median=float(np.median(exact_ranks)),
        exact_rank_range=[int(exact_ranks.min()), int(exact_ranks.max())],
        dense_size_median=float(np.median(dense)),
    )
    print(f"[rankdial] over {summary['n_cores']} cores: KL_rank1 median="
          f"{summary['kl_rank1_median']:.2f} exact_rank median={summary['exact_rank_median']}",
          flush=True)

    out = dict(detailed=detailed, summary=summary, all_cores=stats)
    _dump("exp2_rankdial.json", out)
    # detailed table CSV
    if detailed is not None:
        csv = os.path.join(lib.RESULTS_DIR, "exp2_rankdial_table.csv")
        with open(csv, "w") as fh:
            fh.write("rank,KL,TV,gap_closed,projected,pre_projection_sum,"
                     "projected_negative_mass,min_value_before_projection\n")
            for row in detailed["rows"]:
                fh.write(f"{row['rank']},{row['kl']:.4f},{row['tv']:.4f},"
                         f"{row['gap_closed']:.4f},{row.get('projected', False)},"
                         f"{row.get('pre_projection_sum', float('nan')):.8f},"
                         f"{row.get('projected_negative_mass', float('nan')):.8f},"
                         f"{row.get('min_value_before_projection', float('nan')):.8f}\n")
    return out


# ---------------------------------------------------------------------------------- utils
def _dump(name, obj):
    path = os.path.join(lib.RESULTS_DIR, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    print("wrote", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["treewidth", "corecut", "modular", "rankdial", "all"])
    ap.add_argument("--namespaces", default="cc,mf,bp")
    ap.add_argument("--target_tw", type=int, default=14)
    args = ap.parse_args()

    lib.ensure_results_dir()
    theories = lib.load_theories()
    nss = args.namespaces.split(",")

    if args.part in ("treewidth", "all"):
        run_treewidth(theories, nss)
    if args.part in ("corecut", "all"):
        # cc is already low-tw; core-cut targets mf and bp
        run_corecut(theories, [ns for ns in nss if ns != "cc"], target_tw=args.target_tw)
    if args.part in ("modular", "all"):
        run_modular(theories, nss, target_tw=args.target_tw)
    if args.part in ("rankdial", "all"):
        run_rankdial(theories)


if __name__ == "__main__":
    main()
