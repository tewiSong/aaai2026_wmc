"""Modular exact WMC: cut a small reconvergence core, then exact-compute the rest.

The high treewidth of a GO namespace is carried by a few high-degree reconvergence
hubs. We greedily remove the highest-degree terms in fixed-fraction batches and recompute
the min-fill width of the remainder until it drops below a target (so exact WMC is
feasible on every remaining connected module). The complementary terms then admit exact
junction-tree WMC module by module.

This module provides:
  - `core_cut_curve`: the full trade-off of (fraction cut, max remaining treewidth,
    fraction of terms left exactly computable, time),
  - `select_core_boundary_aware`: greedily cut until every module plus its core boundary
    is low-treewidth,
  - `modular_marginals_with_boundary`: run exact JT marginals over all augmented modules
    while carrying a weighted separator message from the core boundary,
  - `modular_exact_marginals`: an exact disconnected-subtheory reference path, valid for
    whole low-treewidth graphs and core-free runs.

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
from .treewidth import treewidth_min_fill, min_fill_width_bounded


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
    """Exact JT marginals over disconnected non-core components.

    This is exact for the graph it actually calibrates. If `core` is non-empty it drops
    all clauses crossing the cut, so it must not be used as the modular separator-message
    construction from the paper.
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


def augmented_module_graph(
    theory: TruePathTheory,
    core: set,
    comp: set,
    constraint_graph: Optional[nx.Graph] = None,
) -> nx.Graph:
    """Return the graph for one module plus its adjacent core boundary.

    The augmented graph contains module-internal edges and module-boundary edges. Core-core
    edges among boundary atoms are deliberately excluded because their dependence is supplied
    by the separator message, not re-applied inside the module.
    """
    g0 = constraint_graph if constraint_graph is not None else theory.constraint_graph()
    core_set = set(core)
    comp_set = set(comp)
    bnd = set()
    for v in comp_set:
        for w in g0.neighbors(v):
            if w in core_set:
                bnd.add(w)

    aug_nodes = comp_set | bnd
    aug = nx.Graph()
    aug.add_nodes_from(aug_nodes)
    aug.add_edges_from(g0.subgraph(comp_set).edges())
    for v in comp_set:
        for w in g0.neighbors(v):
            if w in bnd:
                aug.add_edge(v, w)
    return aug


def boundary_aware_module_stats(
    theory: TruePathTheory,
    core: set,
    stop_above: Optional[int] = None,
) -> Dict:
    """Measure widths of modules after adding their adjacent core boundary.

    When `stop_above` is set, the scan stops at the first augmented module whose
    treewidth exceeds that value. This is used only to reject an insufficient core early;
    successful reported stats always scan all modules.
    """
    g0 = theory.constraint_graph()
    core_set = set(core)
    noncore = [v for v in theory.atoms if v not in core_set]
    sub = g0.subgraph(noncore)

    max_internal_tw = 0
    max_augmented_tw = 0
    max_boundary_size = 0
    max_module_size = 0
    n_modules = 0
    n_boundary_edges = 0
    witness = None

    for comp in nx.connected_components(sub):
        comp_set = set(comp)
        internal = sub.subgraph(comp_set).copy()
        internal_tw = (
            min_fill_width_bounded(internal, stop_above)
            if stop_above is not None else treewidth_min_fill(internal)
        )
        aug = augmented_module_graph(theory, core_set, comp_set, constraint_graph=g0)
        aug_tw = (
            min_fill_width_bounded(aug, stop_above)
            if stop_above is not None else treewidth_min_fill(aug)
        )
        bnd = set(aug.nodes()) - comp_set

        max_internal_tw = max(max_internal_tw, internal_tw)
        max_augmented_tw = max(max_augmented_tw, aug_tw)
        max_boundary_size = max(max_boundary_size, len(bnd))
        max_module_size = max(max_module_size, len(comp_set))
        n_modules += 1
        for u, w in aug.edges():
            if (u in comp_set and w in bnd) or (w in comp_set and u in bnd):
                n_boundary_edges += 1

        if stop_above is not None and aug_tw > stop_above:
            witness = dict(
                module_size=len(comp_set),
                boundary_size=len(bnd),
                augmented_tw=aug_tw,
                first=sorted(comp_set)[0] if comp_set else None,
            )
            break

    n_total = theory_atoms_count(theory)
    return dict(
        n_modules=n_modules,
        max_internal_tw=max_internal_tw,
        max_augmented_tw=max_augmented_tw,
        max_boundary_size=max_boundary_size,
        max_module_size=max_module_size,
        n_boundary_edges=n_boundary_edges,
        n_terms_exact=len(noncore),
        exact_frac=len(noncore) / n_total,
        core_size=len(core_set),
        core_frac=len(core_set) / n_total,
        stopped_early=witness is not None,
        witness=witness,
    )


def modular_marginals_with_boundary(
    theory: TruePathTheory,
    priors: Dict[object, float],
    core: set,
    boundary_marg: Dict[object, float],
    tw_cap: int = 24,
) -> Tuple[Dict[object, float], Dict]:
    """Correct modular marginals: each module solved EXACTLY conditional on its core
    boundary, the boundary carried by a rank-1 separator message.

    For module M (a connected component of the non-core graph) with boundary core atoms
    B = core atoms adjacent to M, we build a junction tree over M U B, set the unary
    potential of every boundary atom b to its core marginal ``boundary_marg[b]`` (the
    rank-1 / independent separator message, supplied by the soft closure on the core), and
    INCLUDE the boundary clauses linking M to B. Calibrating and reading the module atoms
    then gives P(y_m | core boundary) -- exact within the module, the boundary approximated
    only to the rank of the separator message (rank-1 here; the rank-r dial closes the
    residual, Exp 6 / `rank_dial`). This is the construction the paper describes; it does
    NOT drop the boundary constraints (the earlier severed version did, which is wrong).

    Boundary atoms keep their *core* marginal as a soft external prior and are not reported;
    only module-internal terms are returned. If an augmented (M U B) module exceeds
    ``tw_cap``, the run fails immediately because those terms cannot be claimed exact under
    the requested cap.
    """
    g0 = theory.constraint_graph()
    core_set = set(core)
    noncore = [v for v in theory.atoms if v not in core_set]
    sub = g0.subgraph(noncore)
    gidx = {a: i + 1 for i, a in enumerate(theory.atoms)}
    clauses = theory.clauses()

    marg: Dict[object, float] = {}
    max_tw = 0
    n_modules = 0
    n_boundary_clauses_kept = 0
    t0 = time.time()
    for comp in nx.connected_components(sub):
        comp = set(comp)
        aug = augmented_module_graph(theory, core_set, comp, constraint_graph=g0)
        aug_nodes = set(aug.nodes())
        bnd = aug_nodes - comp
        # priors: module atoms at their prior, boundary atoms at their core marginal
        loc_priors = {}
        for v in aug_nodes:
            loc_priors[v] = boundary_marg[v] if v in core_set else priors[v]
        jt = JunctionTree(aug, gidx)
        if jt.width > tw_cap:
            first = sorted(comp)[0] if comp else None
            raise RuntimeError(
                f"augmented module exceeds tw_cap={tw_cap}: first={first}, "
                f"module_size={len(comp)}, boundary_size={len(bnd)}, width={jt.width}"
            )
        max_tw = max(max_tw, jt.width)
        # Clauses for the module solve: those involving at least one module atom (module-
        # internal + boundary). Core-core clauses among boundary atoms are excluded -- they
        # are summarized by the independent boundary marginals, not re-applied here.
        comp_gidx = {gidx[v] for v in comp}
        mod_clauses = [cl for cl in clauses if any(abs(l) in comp_gidx for l in cl)]
        m = jt.calibrate_marginals(mod_clauses, loc_priors)
        # keep only module-internal atoms (not the borrowed boundary atoms)
        for v in comp:
            if v in m:
                marg[v] = m[v]
        # count boundary clauses actually folded in
        for c, p in theory.nf1:
            if (c in comp and p in bnd) or (p in comp and c in bnd):
                n_boundary_clauses_kept += 1
        n_modules += 1
    dt = time.time() - t0
    n_total = theory_atoms_count(theory)
    stats = dict(
        n_modules=n_modules,
        max_module_tw=max_tw,
        n_terms_exact=len(marg),
        exact_frac=len(marg) / n_total,
        boundary_clauses_kept=n_boundary_clauses_kept,
        time_s=dt,
        core_size=len(core),
        core_frac=len(core) / n_total,
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


def select_core_boundary_aware(
    theory: TruePathTheory,
    target_tw: int = 14,
    batch_frac: float = 0.01,
    max_frac: float = 0.30,
) -> set:
    """Return a greedy core whose augmented modules all satisfy `target_tw`.

    Unlike `select_core`, this checks the actual object used by the modular separator
    construction: every connected non-core module plus its adjacent core boundary. If the
    cap cannot be reached within `max_frac`, the function raises instead of silently
    reporting a severed or over-width modular run.
    """
    g0 = theory.constraint_graph()
    n_total = g0.number_of_nodes()
    scores = _hub_scores(theory)
    ranked = sorted(theory.atoms, key=lambda v: scores[v], reverse=True)
    removed: set = set()
    batch_size = max(1, int(round(batch_frac * n_total)))
    ptr = 0

    while True:
        stats = boundary_aware_module_stats(theory, removed, stop_above=target_tw)
        full = stats
        print(
            f"[boundary-core] cut={len(removed):5d} ({100*len(removed)/n_total:5.2f}%) "
            f"max_aug_tw={full['max_augmented_tw']:4d} modules={full['n_modules']:5d} "
            f"left={full['n_terms_exact']:6d} ({100*full['exact_frac']:.1f}%)",
            flush=True,
        )
        if not stats["stopped_early"]:
            return removed
        if len(removed) / n_total >= max_frac:
            w = stats["witness"]
            raise RuntimeError(
                f"boundary-aware core did not reach tw <= {target_tw} within "
                f"{100*max_frac:.1f}% cut; witness={w}"
            )
        added = 0
        while ptr < len(ranked) and added < batch_size:
            v = ranked[ptr]
            ptr += 1
            if v not in removed:
                removed.add(v)
                added += 1
        if added == 0:
            raise RuntimeError("boundary-aware core selection exhausted all ranked atoms")


def theory_atoms_count(theory: TruePathTheory) -> int:
    return len(theory.atoms)
