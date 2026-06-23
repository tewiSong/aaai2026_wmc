"""#5: run the ACTUAL DeepProbLog system (ExactEngine, SDD knowledge compilation) on the GO
true-path conditioned WMC.

DeepProbLog's exact inference compiles the grounded program to an arithmetic circuit; we encode
the conditioned posterior mu_i = P(y_i, T)/P(T) (T = no true-path edge violated) and read it as
a ratio of two DeepProbLog queries. We verify mu_a = 5/6 on the diamond, then run it on
increasing-treewidth GO subgraphs and on the full namespaces, recording DeepProbLog's own
compile time and where it walls -- a direct test of "exact compilation compiles cc, not mf/bp".
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import warnings

warnings.filterwarnings("ignore")

import networkx as nx

import lib
from truepath import treewidth as twmod


def _program(atoms, edges, priors, target):
    lines = [f"{priors[a]:.6f}::y('{a}')." for a in atoms]
    lines += [f"edge('{c}','{p}')." for c, p in edges]
    lines.append("viol :- edge(C,P), y(C), \\+ y(P).")
    lines.append("ok :- \\+ viol.")
    lines.append(f"qa :- y('{target}'), ok.")
    return "\n".join(lines)


def _dpl_worker(atoms, edges, priors, target, q):
    from deepproblog.model import Model
    from deepproblog.engines import ExactEngine
    from deepproblog.query import Query
    from problog.logic import Term
    m = Model(_program(atoms, edges, priors, target), [], load=False)
    m.set_engine(ExactEngine(m))
    t0 = time.time()
    r = m.solve([Query(Term("ok")), Query(Term("qa"))])
    pok = float(list(r[0].result.values())[0])
    pqa = float(list(r[1].result.values())[0])
    if pok <= 0.0:
        raise FloatingPointError("DeepProbLog returned zero mass for ok")
    q.put((pqa / pok, time.time() - t0, float(r[0].compile_time + r[1].compile_time)))


def run_dpl(atoms, edges, priors, target, timeout_s=300):
    q = mp.Queue()
    p = mp.Process(target=_dpl_worker, args=(atoms, edges, priors, target, q))
    p.start(); p.join(timeout_s)
    if p.is_alive():
        p.terminate(); p.join(); return None
    if q.empty():
        raise RuntimeError("DeepProbLog worker exited without returning a result")
    return q.get()


def main():
    lib.ensure_results_dir()
    out = {}

    # 1) diamond: must give mu_a = 5/6
    res = run_dpl(["a", "b", "c", "d"],
                  [("d", "b"), ("d", "c"), ("b", "a"), ("c", "a")],
                  {x: 0.5 for x in "abcd"}, "a", timeout_s=60)
    out["diamond_mu_a"] = res[0] if res else None
    print(f"[deepproblog] diamond mu_a = {res[0]:.4f} (exact 5/6={5/6:.4f}), {res[1]:.2f}s", flush=True)

    theories = lib.load_theories()
    # 2) increasing-treewidth subgraphs of bp: DeepProbLog ExactEngine compile time vs tw
    th = theories["bp"]
    priors = lib.make_priors(th, seed=0)
    g = th.constraint_graph()
    deg = dict(g.degree())
    ranked = sorted(th.atoms, key=lambda v: deg.get(v, 0), reverse=True)
    parents = {}
    for c, p in th.nf1:
        parents.setdefault(c, []).append(p)
    rows = []
    best_tw = -1
    for size in [40, 120, 300, 700, 1500, 3000]:
        nodes = set()
        for v in ranked:
            if len(nodes) >= size:
                break
            nodes.add(v)
            for p in parents.get(v, []):
                if len(nodes) < size:
                    nodes.add(p)
        sub = g.subgraph(nodes)
        comp = max(nx.connected_components(sub), key=len)
        sg = g.subgraph(comp)
        tw = twmod.treewidth_min_fill(sg)
        if tw <= best_tw:
            continue
        best_tw = tw
        atoms = sorted(comp)
        edges = [(c, p) for c, p in th.nf1 if c in set(atoms) and p in set(atoms)]
        target = max(atoms, key=lambda a: deg.get(a, 0))
        res = run_dpl(atoms, edges, priors, target, timeout_s=300)
        if res is None:
            rows.append(dict(n=len(atoms), treewidth=tw, walled=True, dpl_time_s=None))
            print(f"[deepproblog] subgraph n={len(atoms)} tw={tw}: WALL (>300s)", flush=True)
            break
        rows.append(dict(n=len(atoms), treewidth=tw, walled=False,
                         dpl_time_s=round(res[1], 2), compile_s=round(res[2], 2)))
        print(f"[deepproblog] subgraph n={len(atoms)} tw={tw}: {res[1]:.2f}s "
              f"(compile {res[2]:.2f}s)", flush=True)
    out["bp_subgraph_scaling"] = rows

    # 3) full namespaces
    ns_res = {}
    for ns in ["cc", "mf"]:
        th = theories[ns]
        pri = lib.make_priors(th, seed=0)
        atoms = list(th.atoms)
        edges = th.nf1
        target = atoms[0]
        res = run_dpl(atoms, edges, pri, target, timeout_s=300)
        ns_res[ns] = dict(treewidth=twmod.treewidth_min_fill(th.constraint_graph()),
                          ok=res is not None, time_s=(round(res[1], 1) if res else None))
        print(f"[deepproblog] full {ns}: {'%.1fs' % res[1] if res else 'WALL (>300s)'}", flush=True)
    out["namespaces"] = ns_res

    path = os.path.join(lib.RESULTS_DIR, "exp3e_real_deepproblog.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
