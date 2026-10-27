"""P4.0  Does exact ontology conditioning help real predictions? DeepGOPlus 1.0.6 CNN scores on
its 3,874 held-out test proteins, GO cc and GO mf vocabulary theories (exactly solvable).

Purpose (Codex audit, point 5): P1.4a shows conditioning moves dense scores a lot and BP has
sizeable errors, but neither says whether the conditioned posterior is better against true
labels. Methods compared on the same scores and labels:
  raw          CNN scores
  max_closure  parent score = max(own, descendants)   (common hierarchy post-processing)
  exact_wmc    exact conditioned marginals (d4 circuit compiled once)
  bp           loopy BP marginals of the same model
Metrics: protein-centric Fmax (CAFA), micro AUPR, Brier, ECE (15 bins), hierarchy
violations; paired bootstrap (1,000 resamples of proteins) for differences vs exact_wmc.

Output: results/p4_0_downstream_dgp_<ns>.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from wmc_common import DATA_DIR, RESULTS_DIR, load_theory, write_dimacs
from wmc_inference import Circuit, compile_with_d4, loopy_bp, violations


def max_closure(S, edges, n):
    children = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=int)
    parents = [[] for _ in range(n)]
    for c, q in edges:
        children[q].append(c); parents[c].append(q); indeg[q] += 1
    order, stack = [], [v for v in range(n) if indeg[v] == 0]
    while stack:  # leaves first
        v = stack.pop(); order.append(v)
        for u in parents[v]:
            indeg[u] -= 1
            if indeg[u] == 0:
                stack.append(u)
    out = S.copy()
    for v in order:
        for u in parents[v]:
            out[:, u] = np.maximum(out[:, u], out[:, v])
    return out


def fmax(P, Y, thresholds=np.linspace(0.01, 0.99, 99)):
    best = 0.0
    for t in thresholds:
        pred = P >= t
        has = pred.any(1)
        if not has.any():
            continue
        tp = (pred & (Y == 1)).sum(1)
        prec = (tp[has] / pred[has].sum(1)).mean()
        rec = (tp / np.maximum(Y.sum(1), 1)).mean()
        if prec + rec > 0:
            best = max(best, 2 * prec * rec / (prec + rec))
    return best


def aupr(P, Y):
    s, y = P.ravel(), Y.ravel()
    o = np.argsort(-s, kind="stable")
    y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / (tp + fp); rec = tp / max(y.sum(), 1)
    return float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))


def ece(P, Y, bins=15):
    s, y = P.ravel(), Y.ravel()
    idx = np.minimum((s * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(s[m].mean() - y[m].mean())
    return tot / len(s)


def metrics(P, Y, edges):
    return dict(fmax=fmax(P, Y), aupr=aupr(P, Y), brier=float(np.mean((P - Y) ** 2)), ece=ece(P, Y),
                violation_rate=float(np.mean(P[:, edges[:, 0]] > P[:, edges[:, 1]] + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    name = f"dgp_{args.ns}"
    ids, edges = load_theory(name)
    n = len(ids)
    z = np.load(DATA_DIR / "scores" / f"deepgoplus_{args.ns}.npz")
    S, Y = z["scores"].astype(np.float64), z["labels"].astype(np.int8)
    keep = Y.sum(1) > 0                       # proteins with at least one label in this namespace
    S, Y = S[keep], Y[keep]
    with tempfile.TemporaryDirectory() as td:
        cnf, nnf = Path(td) / "t.cnf", Path(td) / "t.nnf"
        write_dimacs(cnf, n, edges)
        compile_with_d4(cnf, nnf)
        circ = Circuit(nnf, n)
        E = np.stack([circ.marginals(S[i])[0] for i in range(len(S))])
    B = np.stack([loopy_bp(n, edges, S[i])["mu"] for i in range(len(S))])
    M = max_closure(S, edges, n)
    outs = dict(raw=S, max_closure=M, exact_wmc=E, bp=B)
    res = dict(theory=name, terms=n, proteins=int(len(S)),
               metrics={k: metrics(v, Y, edges) for k, v in outs.items()})
    rng = np.random.default_rng(0)
    boot = {k: {m: [] for m in ("fmax", "aupr", "brier", "ece")} for k in outs if k != "exact_wmc"}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(S), len(S))
        me = metrics(E[idx], Y[idx], edges)
        for k in boot:
            mk = metrics(outs[k][idx], Y[idx], edges)
            for m in boot[k]:
                boot[k][m].append(me[m] - mk[m])
    res["exact_minus_other_95ci"] = {k: {m: np.quantile(v, [0.025, 0.5, 0.975]).tolist() for m, v in d.items()}
                                     for k, d in boot.items()}
    print(json.dumps(res, indent=1))
    json.dump(res, open(RESULTS_DIR / f"p4_0_downstream_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
