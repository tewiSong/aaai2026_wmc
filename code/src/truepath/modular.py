"""Modular exact WMC: cut a small reconvergence core, then exact-compute the rest.

The high treewidth of a GO namespace is carried by a few high-degree reconvergence
hubs. We greedily remove the highest-degree terms in fixed-fraction batches and recompute
the min-fill treewidth of the remainder until it drops below a target (so exact WMC is
feasible on every remaining connected module). The complementary terms then admit exact
junction-tree WMC module by module.

This module provides:
  - `core_cut_curve`: the full trade-off of (fraction cut, max remaining treewidth,
    fraction of terms left exactly computable, time),
  - `modular_exact_marginals`: run exact JT marginals over all modules of the core-cut
    remainder, returning marginals plus per-module statistics.

Degree here is the constraint-graph degree; ties are broken by NF1 in-degree (the number
of parents, i.e. how reconvergent the node is), matching the multi-parent hub structure.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .graph import TruePathTheory
from .junction_tree import JunctionTree
from .treewidth import treewidth_min_fill


def _hub_scores(theory: TruePathTheory) -> Dict[object, Tuple[int, int]]:
    """Score each atom by (constraint-graph degree, nf1 in-degree). Higher = more central."""
    g = theory.constraint_graph()
    indeg = Counter()
    for c, _p in theory.nf1:
        indeg[c] += 1  # number of parents of c
    # also count being a parent target (out usage) via nf2/nf3/nf4 implicitly via degree
    scores = {}
    for v in theory.atoms:
        scores[v] = (g.degree(v) if v in g else 0, indeg[v])
    return scores


def core_cut_curve(
    theory: TruePathTheory,
    target_tw: int = 14,
    batch_frac: float = 0.01,
    max_frac: float = 0.30,
) -> List[Dict]:
    """Greedily remove highest-degree terms in `batch_frac` batches.

    Returns a list of records, one per batch (including batch 0 = nothing removed),
    each with: cut_count, cut_frac, max_tw, exact_frac, n_modules, time_s.
    Stops once max remaining treewidth <= target_tw or `max_frac` removed.
    """
    g0 = theory.constraint_graph()
    n_total = g0.number_of_nodes()
    scores = _hub_scores(theory)
    ranked = sorted(theory.atoms, key=lambda v: scores[v], reverse=True)

    records: List[Dict] = []
    removed: set = set()
    batch_size = max(1, int(round(batch_frac * n_total)))
    ptr = 0
    while True:
        g = g0.subgraph([v for v in theory.atoms if v not in removed])
        t0 = time.time()
        tw = treewidth_min_fill(g)
        dt = time.time() - t0
        n_left = g.number_of_nodes()
        n_modules = nx.number_connected_components(g)
        rec = dict(
            cut_count=len(removed),
            cut_frac=len(removed) / n_total,
            max_tw=tw,
            exact_frac=n_left / n_total,
            n_modules=n_modules,
            tw_time_s=dt,
        )
        records.append(rec)
        print(f"[core-cut] cut={len(removed):5d} ({100*len(removed)/n_total:5.2f}%) "
              f"max_tw={tw:4d} modules={n_modules:5d} left={n_left:6d} ({100*rec['exact_frac']:.1f}%) "
              f"tw_time={dt:.1f}s", flush=True)
        if tw <= target_tw:
            break
        if len(removed) / n_total >= max_frac:
            break
        # remove next batch of highest-scoring still-present nodes
        added = 0
        while ptr < len(ranked) and added < batch_size:
            v = ranked[ptr]
            ptr += 1
            if v not in removed:
                removed.add(v)
                added += 1
        if added == 0:
            break
    return records


def modular_exact_marginals(
    theory: TruePathTheory,
    priors: Dict[object, float],
    core: set,
) -> Tuple[Dict[object, float], Dict]:
    """Exact JT marginals over every module (connected component of the non-core graph).

    Returns (marginals over non-core terms, stats dict with module count, max module tw,
    total time).
    """
    g0 = theory.constraint_graph()
    sub_nodes = [v for v in theory.atoms if v not in core]
    g = g0.subgraph(sub_nodes)
    gidx = {a: i + 1 for i, a in enumerate(theory.atoms)}
    clauses = theory.clauses()

    marg: Dict[object, float] = {}
    max_tw = 0
    n_modules = 0
    t0 = time.time()
    for comp in nx.connected_components(g):
        sub = g.subgraph(comp).copy()
        jt = JunctionTree(sub, gidx)
        max_tw = max(max_tw, jt.width)
        m = jt.calibrate_marginals(clauses, priors)
        marg.update(m)
        n_modules += 1
    dt = time.time() - t0
    stats = dict(
        n_modules=n_modules,
        max_module_tw=max_tw,
        n_terms_exact=len(marg),
        exact_frac=len(marg) / theory_atoms_count(theory),
        time_s=dt,
        core_size=len(core),
        core_frac=len(core) / theory_atoms_count(theory),
    )
    return marg, stats


def select_core(theory: TruePathTheory, target_tw: int = 14, max_frac: float = 0.30) -> set:
    """Return the set of core terms whose removal drops the remainder treewidth to
    `target_tw`, using the same greedy degree ranking as `core_cut_curve`."""
    g0 = theory.constraint_graph()
    n_total = g0.number_of_nodes()
    scores = _hub_scores(theory)
    ranked = sorted(theory.atoms, key=lambda v: scores[v], reverse=True)
    removed: set = set()
    batch_size = max(1, int(round(0.01 * n_total)))
    ptr = 0
    while True:
        g = g0.subgraph([v for v in theory.atoms if v not in removed])
        tw = treewidth_min_fill(g)
        if tw <= target_tw or len(removed) / n_total >= max_frac:
            break
        added = 0
        while ptr < len(ranked) and added < batch_size:
            v = ranked[ptr]; ptr += 1
            if v not in removed:
                removed.add(v); added += 1
        if added == 0:
            break
    return removed


def theory_atoms_count(theory: TruePathTheory) -> int:
    return len(theory.atoms)
