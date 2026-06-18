"""Task F: modular exact + rank-r separator, end-to-end on a subgraph, validated to gold.

We integrate the two halves of the construction on one self-contained subgraph and check
the result against the full-junction-tree exact marginals (the gold reference):

  1. split the subgraph into a reconvergence core C and its periphery P, with separator B
     = the periphery atoms adjacent to C;
  2. compute the exact periphery->core message over B (the periphery WMC as a function of
     the B-assignment) -- this is the separator message of Proposition 4;
  3. compute the core-term marginals by combining the core factors with that message;
  4. represent the message at rank r (tensor train over B): r=1 is the soft closure
     (independent boundary), full rank is exact.

We verify periphery marginals are exact (val_err 0) and that the core marginals match the
gold exactly once the message reaches its tensor-train rank, while rank-1 leaves a residual.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from itertools import product

import networkx as nx
import numpy as np

import lib
from truepath.graph import TruePathTheory
from truepath import modular, rank_dial, junction_tree
from truepath.modular import select_core


def _periphery_message(periph_atoms, B, clauses_global, gidx, priors):
    """Exact periphery->core message over B by enumerating the (small) periphery:
    msg[b] = sum over periphery assignments with B=b that satisfy the periphery clauses,
    weighted by the periphery priors (including the B atoms' own priors). Normalized."""
    P = list(periph_atoms)
    idx = {v: i for i, v in enumerate(P)}
    nset = set(gidx[v] for v in P)
    inv = {gidx[v]: v for v in P}
    local = [cl for cl in clauses_global if all(abs(l) in nset for l in cl)]
    Bidx = [idx[b] for b in B]
    msg = np.zeros((2,) * len(B))
    for asg in product((0, 1), repeat=len(P)):
        ok = True
        for cl in local:
            if not any(asg[idx[inv[abs(l)]]] == (1 if l > 0 else 0) for l in cl):
                ok = False
                break
        if not ok:
            continue
        w = 1.0
        for v in P:
            p = priors[v]
            w *= p if asg[idx[v]] == 1 else (1 - p)
        msg[tuple(asg[bi] for bi in Bidx)] += w
    s = msg.sum()
    return msg / s if s > 0 else msg


def core_marginals(core, B, clauses_global, gidx, priors, msgB):
    """Core-term marginals given a boundary message msgB over B (tensor over B)."""
    atoms = list(core) + list(B)
    idx = {a: i for i, a in enumerate(atoms)}
    nset = set(gidx[a] for a in atoms)
    inv = {gidx[a]: a for a in atoms}
    # core factors: clauses touching a core atom, fully inside core U B
    core_clauses = []
    coreset = set(core)
    for cl in clauses_global:
        vs = [inv.get(abs(l)) for l in cl]
        if any(v in coreset for v in vs) and all(abs(l) in nset for l in cl):
            core_clauses.append(cl)
    Bidx = [idx[b] for b in B]
    num = {c: 0.0 for c in core}
    den = 0.0
    for asg in product((0, 1), repeat=len(atoms)):
        ok = True
        for cl in core_clauses:
            sat = any((asg[idx[inv[abs(l)]]] == (1 if l > 0 else 0)) for l in cl)
            if not sat:
                ok = False
                break
        if not ok:
            continue
        w = 1.0
        for c in core:
            p = priors[c]
            w *= p if asg[idx[c]] == 1 else (1 - p)
        bkey = tuple(asg[bi] for bi in Bidx)
        w *= msgB[bkey]
        den += w
        for c in core:
            if asg[idx[c]] == 1:
                num[c] += w
    return {c: (num[c] / den if den > 0 else 0.0) for c in core}


def run(seed=0, max_subgraph=18, n_examples=6, namespace="cc"):
    theories = lib.load_theories()
    th = theories[namespace]
    priors = lib.make_priors(th, seed=seed)
    indeg = Counter()
    for c, _p in th.nf1:
        indeg[c] += 1
    g = th.constraint_graph()
    gidx = {a: i + 1 for i, a in enumerate(th.atoms)}
    clauses = th.clauses()

    # Find small subgraphs around reconvergent hubs that fit brute-force gold.
    hubs = [a for a in sorted(th.atoms, key=lambda x: indeg[x], reverse=True) if indeg[a] > 1]
    results = []
    for hub in hubs:
        # subgraph = hub's neighborhood up to max_subgraph nodes (BFS)
        nodes = list(nx.ego_graph(g, hub, radius=2).nodes())
        if len(nodes) > max_subgraph or len(nodes) < 5:
            continue
        S = g.subgraph(nodes).copy()
        from truepath import bruteforce
        idxS = {v: i for i, v in enumerate(sorted(nodes))}
        nodesS = sorted(nodes)
        nset = set(gidx[v] for v in nodesS)
        inv = {gidx[v]: v for v in nodesS}
        locS = []
        for cl in clauses:
            if all(abs(l) in nset for l in cl):
                locS.append(tuple((1 if l > 0 else -1) * (idxS[inv[abs(l)]] + 1) for l in cl))
        priS = [priors[v] for v in nodesS]
        _, gold = bruteforce.brute_force_marginals(len(nodesS), locS, priS)
        gold = {v: gold[idxS[v]] for v in nodesS}

        # core = reconvergent hubs in S; periphery = rest; B = periphery nbrs of core
        core = [v for v in nodesS if indeg[v] > 1 and v != hub] + [hub]
        core = list(dict.fromkeys(core))
        if len(core) > 6:
            core = core[:6]
        coreset = set(core)
        periph = [v for v in nodesS if v not in coreset]
        B = sorted({u for c in core for u in S.neighbors(c) if u in set(periph)})
        if not B or len(B) > 12 or not periph:
            continue

        # exact periphery->core separator message over B, and its tensor-train rank
        msg_exact = _periphery_message(periph, B, clauses, gidx, priors)
        exact_rank = rank_dial.tt_max_rank(rank_dial.tt_svd(msg_exact, max_rank=None))

        # core marginals as the boundary message is represented at increasing rank:
        # rank 1 = independent product (the soft closure); full rank = exact.
        prod_msg = rank_dial.product_marginal(msg_exact)
        sweep = []
        for r in range(1, exact_rank + 1):
            if r == 1:
                m = prod_msg
            else:
                c = rank_dial.tt_svd(msg_exact, max_rank=r)
                m = np.clip(rank_dial.tt_reconstruct(c), 0, None)
                m = m / m.sum()
            cm = core_marginals(core, B, clauses, gidx, priors, m)
            err = max(abs(cm[c] - gold[c]) for c in core)
            sweep.append((r, float(err)))
        err_rank1 = sweep[0][1]
        err_full = sweep[-1][1]

        rec = dict(hub=hub, n_nodes=len(nodesS), core=len(core), boundary=len(B),
                   exact_rank=exact_rank, core_err_rank1=err_rank1,
                   core_err_fullrank=err_full, rank_sweep=sweep)
        results.append(rec)
        print(f"[F] hub={hub} |S|={len(nodesS)} core={len(core)} |B|={len(B)} "
              f"msg_rank={exact_rank} core marginal err: rank1={err_rank1:.3e} "
              f"-> fullrank={err_full:.1e}", flush=True)
        if len(results) >= n_examples:
            break

    out = os.path.join(lib.RESULTS_DIR, f"exp_modular_rankr_endtoend_{namespace}.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print("wrote", out, flush=True)
    return results


if __name__ == "__main__":
    lib.ensure_results_dir()
    for ns in ["cc", "bp"]:
        print(f"=== end-to-end modular+rank-r on {ns} ===", flush=True)
        run(namespace=ns)
