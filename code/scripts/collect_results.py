"""Collect all experiment outputs in results/ into a single human-readable summary."""

from __future__ import annotations

import json
import os

import lib


def _load(name):
    path = os.path.join(lib.RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def main():
    print("=" * 72)
    print("EXPERIMENT SUMMARY")
    print("=" * 72)

    d = _load("exp0_dataset_stats.json")
    if d:
        print("\n[Exp0] Dataset")
        for r in d:
            print(f"  {r['namespace']}: terms={r['n_terms']} edges={r['n_edges']} "
                  f"multi-parent={r['multi_parent']} ({100*r['multi_parent_frac']:.0f}%) "
                  f"max_in_degree={r['max_in_degree']}")

    b = _load("exp1_battery.json")
    if b:
        print("\n[Exp1] Prior battery (BP vs exact)")
        for k, v in b.items():
            print(f"  {k}: max_err={v['max_err_over_grid']:.1e} "
                  f"({'EXACT' if v['exact_on_grid'] else 'INEXACT'})")
    c = _load("exp1_cc.json")
    if c:
        print("\n[Exp1] cc soft closure vs exact WMC")
        print(f"  exact JT {c['exact_time_s']}s; BP {c['bp_time_s']}s")
        print(f"  mean={c['mean_err']:.2e} p95={c['p95_err']:.2e} max={c['max_err']:.2e}")
        print(f"  multi-parent/single-parent error ratio = {c['localization_ratio']:.1f}x")

    tw = _load("exp2_treewidth.json")
    if tw:
        print("\n[Exp2] Treewidth")
        for ns, v in tw.items():
            print(f"  {ns}: tw={v['treewidth']} (terms={v['n_terms']}, {v['time_s']}s)")
    cc = _load("exp2_corecut.json")
    if cc:
        print("\n[Exp2] Core-cut (final batch)")
        for ns, recs in cc.items():
            last = recs[-1]
            print(f"  {ns}: cut {100*last['cut_frac']:.1f}% -> max_tw {last['max_tw']}, "
                  f"{100*last['exact_frac']:.1f}% exactly computable")
    mod = _load("exp2_modular.json")
    if mod:
        print("\n[Exp2] Modular exact WMC")
        for ns, v in mod.items():
            print(f"  {ns}: core {100*v['core_frac']:.1f}% modules={v['n_modules']} "
                  f"max_tw={v['max_module_tw']} exact={100*v['exact_frac']:.1f}% "
                  f"time={v['wall_time_s']}s val_err={v['validation_max_abs_err']:.1e}")
    rd = _load("exp2_rankdial.json")
    if rd and rd.get("detailed"):
        print("\n[Exp2] Rank-r dial (detailed core)")
        det = rd["detailed"]
        print(f"  w={det['w']} models={det['n_models']} exact_rank={det['exact_rank']} "
              f"KL_rank1={det['kl_rank1']:.2f}")
        for row in det["rows"]:
            print(f"    rank {row['rank']:2d}: KL={row['kl']:.3f} TV={row['tv']:.3f} "
                  f"gap_closed={row['gap_closed']:.0%}")
        s = rd["summary"]
        print(f"  over {s['n_cores']} cores: KL_rank1 median={s['kl_rank1_median']:.2f} "
              f"exact_rank median={s['exact_rank_median']}")

    cw = _load("exp3a_compile_wall.json")
    if cw:
        print("\n[Exp3a] Exact WMC cost vs treewidth (junction-tree 2^tw clique table)")
        for r in cw["scaling"]:
            if r["walled"]:
                print(f"  n={r['n']} tw={r['treewidth']} exact WMC: WALL (2^tw clique)")
            else:
                print(f"  n={r['n']} tw={r['treewidth']} max-clique=2^{r.get('max_clique','?')} "
                      f"exact={r['exact_s']}s")
        for ns, v in cw.get("namespaces", {}).items():
            s = f"{v['exact_s']}s" if v["monolithic_exact_ok"] else "WALL (2^tw)"
            print(f"  {ns}: tw={v['treewidth']} monolithic exact WMC = {s}")
    sk = _load("exp3b_scallop_topk.json")
    if sk:
        print("\n[Exp3b] Scallop top-k drift")
        fn = sk["four_node"]
        print(f"  four-node exact={fn['exact']}: " +
              " ".join(f"k{k}={v:.3f}" for k, v in fn["topk"].items()))
        bn = sk["bp_neighborhoods"]["summary"]
        print("  bp max-drift: " + " ".join(f"k{k}={bn[k]['max_drift']:.3f}" for k in bn))

    cch = _load("exp3c_carrychain.json")
    if cch:
        print("\n[Exp3c] Carry-chain: exact-IA vs Monte-Carlo (learned-approximator) WMC")
        agg = cch["aggregate"]
        # support both flat {N:..} and nested {mode:{N:..}}
        if agg and isinstance(next(iter(agg.values())), dict) and \
           all(isinstance(v, dict) and "mean" not in v for v in agg.values()):
            for mode, byN in agg.items():
                line = ", ".join(f"N={N}:{v['mean']:.3f}" for N, v in byN.items())
                print(f"  {mode}: {line}")
        else:
            for N, v in agg.items():
                print(f"  N={N}: sum-acc {v['mean']:.3f} +/- {v['std']:.3f}")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
