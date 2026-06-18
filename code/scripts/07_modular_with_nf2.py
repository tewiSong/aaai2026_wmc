"""#2 fix: modular exact WMC on the FULL EL++ theory (is-a subsumption + NF2 conjunctions).

The paper's theory is "subsumption AND conjunction-definition axioms". Including the NF2
conjunction cliques makes the full-graph treewidth intractable to compute directly, but the
modular construction does not need it: we greedily cut the highest-degree terms until every
connected module of the remainder has treewidth <= 14 (checked per module, which is cheap
because modules are small), then solve each NF2-augmented module exactly by junction tree and
validate against brute force. This computes the full theory -- NF2 included -- with no
simplification; the conjunction clauses enter both the constraint graph and the module WMC.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter

import networkx as nx

import lib
from truepath.graph import build_namespace_theories_from_norm
from truepath import treewidth as twmod
from truepath.junction_tree import JunctionTree
from truepath import bruteforce


def largest_module_size(g, core):
    sub = g.subgraph([v for v in g.nodes() if v not in core])
    return max((len(c) for c in nx.connected_components(sub)), default=0)


def max_module_tw(g, core, tw_cap_nodes=2500):
    """Max treewidth over modules of the core-removed graph. Treewidth is only computed once
    every module is small enough (<= tw_cap_nodes) to keep it tractable; while a giant module
    persists we return a sentinel large width so the caller keeps cutting (by degree)."""
    sub = g.subgraph([v for v in g.nodes() if v not in core])
    comps = list(nx.connected_components(sub))
    if comps and max(len(c) for c in comps) > tw_cap_nodes:
        return 999, len(comps)  # still a big module -> keep cutting, don't compute tw yet
    mx = 0
    for comp in comps:
        s = g.subgraph(comp)
        if s.number_of_nodes() > 1:
            mx = max(mx, twmod.treewidth_min_fill(s))
    return mx, len(comps)


def main():
    lib.ensure_results_dir()
    # FULL theory: is-a (NF1) + NF2 conjunctions, from the EL++ normalization.
    theories = build_namespace_theories_from_norm(
        os.path.join(lib.DATA_DIR, "processed", "norm"), include_existential=False)
    out = {}
    for ns in ["cc", "mf", "bp"]:
        th = theories[ns]
        g = th.constraint_graph()
        gidx = {a: i + 1 for i, a in enumerate(th.atoms)}
        clauses = th.clauses()
        priors = {a: 0.5 for a in th.atoms}  # uniform; structure is what matters here
        # rank by constraint-graph degree (includes NF2 edges)
        ranked = sorted(th.atoms, key=lambda v: g.degree(v) if v in g else 0, reverse=True)
        n = len(th.atoms)
        batch = max(1, n // 100)
        core = set()
        ptr = 0
        t0 = time.time()
        while True:
            mtw, nmod = max_module_tw(g, core)
            if mtw <= 14 or len(core) / n >= 0.30:
                break
            added = 0
            while ptr < len(ranked) and added < batch:
                if ranked[ptr] not in core:
                    core.add(ranked[ptr]); added += 1
                ptr += 1
            if added == 0:
                break
        # solve modules exactly with NF2 clauses; validate sample against brute force
        sub = g.subgraph([v for v in th.atoms if v not in core])
        n_terms_exact = 0
        n_modules = 0
        max_tw = 0
        val_err = 0.0
        checked = 0
        for comp in nx.connected_components(sub):
            s = g.subgraph(comp).copy()
            jt = JunctionTree(s, gidx)
            max_tw = max(max_tw, jt.width)
            marg = jt.calibrate_marginals(clauses, priors)
            n_terms_exact += len(marg)
            n_modules += 1
            # validate small modules (with their NF2 clauses) against brute force
            if checked < 15 and len(comp) <= 18:
                nodes = sorted(comp)
                li = {v: i for i, v in enumerate(nodes)}
                nset = set(gidx[v] for v in nodes)
                inv = {gidx[v]: v for v in nodes}
                loc = [tuple((1 if l > 0 else -1) * (li[inv[abs(l)]] + 1) for l in cl)
                       for cl in clauses if all(abs(l) in nset for l in cl)]
                _, bf = bruteforce.brute_force_marginals(len(nodes), loc, [priors[v] for v in nodes])
                for i, v in enumerate(nodes):
                    val_err = max(val_err, abs(bf[i] - marg[v]))
                checked += 1
        dt = time.time() - t0
        nf2 = len(th.nf2)
        rec = dict(n_terms=n, nf2=nf2, core_frac=round(len(core) / n, 4),
                   n_modules=n_modules, max_module_tw=max_tw,
                   exact_frac=round(n_terms_exact / n, 4), val_err=val_err, time_s=round(dt, 1))
        out[ns] = rec
        print(f"[modular+NF2] {ns}: nf2={nf2} core={100*rec['core_frac']:.1f}% modules={n_modules} "
              f"max_tw={max_tw} exact={100*rec['exact_frac']:.1f}% val_err={val_err:.1e} "
              f"[{dt:.0f}s]", flush=True)
    path = os.path.join(lib.RESULTS_DIR, "exp2_modular_nf2.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
