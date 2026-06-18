"""#5: run the ACTUAL Scallop system (scallopy, top-k-proofs provenance) on the GO true-path
inference, confirming it matches our exact-semantics top-k reference (scripts/03b).

Scallop is built from source (Rust nightly) and installed as `scallopy`. We encode the
true-path rule  h(P) :- sub(C,P), h(C)  with probabilistic base facts and read the marginal of
a reconvergent target under top-k-proofs at several k.
"""

from __future__ import annotations

import json
import os

import scallopy

import lib


def scallop_marginal(sub_edges, base_probs, target, k):
    ctx = scallopy.ScallopContext(provenance="topkproofs", k=k)
    ctx.add_relation("sub", (str, str))
    ctx.add_facts("sub", [(None, e) for e in sub_edges])
    ctx.add_relation("x", (str,))
    ctx.add_facts("x", [(p, (t,)) for t, p in base_probs.items()])
    ctx.add_rule("h(X) = x(X)")
    ctx.add_rule("h(P) = sub(C,P), h(C)")
    ctx.run()
    d = {t: p for p, t in ctx.relation("h")}
    return d.get((target,))


def main():
    lib.ensure_results_dir()
    out = {}

    # 1) the four-node reconvergence (exact 0.9375)
    sub = [("d", "b"), ("d", "c"), ("b", "a"), ("c", "a")]
    bp = {t: 0.5 for t in "abcd"}
    four = {str(k): scallop_marginal(sub, bp, "a", k) for k in [1, 2, 3, 5, 10]}
    out["four_node"] = four
    print("[real-scallop] four-node P(h(a)):",
          " ".join(f"k{k}={v:.4f}" for k, v in four.items()), flush=True)

    # 2) a real bp reconvergence neighborhood from the true-path edges
    th = lib.load_theories()["bp"]
    priors = lib.make_priors(th, seed=0)
    # pick a multi-parent hub and take its upper cone (<= 14 terms) as the sub-program
    from collections import Counter
    indeg = Counter()
    for c, p in th.nf1:
        indeg[c] += 1
    parents = {}
    for c, p in th.nf1:
        parents.setdefault(c, []).append(p)
    hub = max((a for a in th.atoms if indeg[a] > 1), key=lambda a: indeg[a])
    nodes = [hub]; seen = {hub}; i = 0
    while i < len(nodes) and len(nodes) < 14:
        for par in parents.get(nodes[i], []):
            if par not in seen and len(nodes) < 14:
                seen.add(par); nodes.append(par)
        i += 1
    nset = set(nodes)
    edges = [(c, p) for c, p in th.nf1 if c in nset and p in nset]
    bprob = {a: priors[a] for a in nodes}
    # exact provenance reference: P(h(t)) = 1 - prod over t and descendants of (1 - p)
    desc = {a: set() for a in nodes}
    children = {}
    for c, p in edges:
        children.setdefault(p, []).append(c)
    def descendants(r):
        st = [r]; s = set()
        while st:
            u = st.pop()
            for ch in children.get(u, []):
                if ch not in s:
                    s.add(ch); st.append(ch)
        return s
    import numpy as np
    nb = {}
    for k in [1, 3, 10, 50]:
        vals = []
        for t in nodes:
            facts = sorted([bprob[s] for s in ([t] + list(descendants(t)))], reverse=True)
            exact = 1 - np.prod([1 - p for p in facts])
            sc = scallop_marginal(edges, bprob, t, k)
            if sc is not None:
                vals.append(abs(exact - sc))
        nb[str(k)] = float(max(vals)) if vals else None
    out["bp_neighborhood_maxdrift"] = nb
    print(f"[real-scallop] bp neighborhood (hub {hub}, {len(nodes)} terms) max drift vs exact:",
          " ".join(f"k{k}={v:.4f}" for k, v in nb.items()), flush=True)

    path = os.path.join(lib.RESULTS_DIR, "exp3d_real_scallop.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
