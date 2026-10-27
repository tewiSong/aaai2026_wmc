"""P4.2  Why does product conditioning hurt? Depth-stratified calibration of raw vs exact-WMC
marginals (DeepGOPlus CNN and fused inputs, GO cc / GO mf vocabulary theories).

For every (protein, term) pair, bucket by term depth (shortest is-a distance to the namespace
root) and compare mean raw score, mean exact conditioned marginal, and the observed label
frequency. If raw scores are calibrated marginals, conditioning should inflate shallow terms
and deflate deep ones (single-edge identity: for c => a with scores c, a,
P_T(a) = a / (1 - c(1-a)) >= a and P_T(c) = a c / (1 - c(1-a)) <= c).
Output: results/p4_2_mechanism_<ns>_<input>.json
"""
import json, sys, tempfile
from collections import deque
from pathlib import Path
import numpy as np
from wmc_common import DATA_DIR, RESULTS_DIR, load_theory, write_dimacs
from wmc_inference import Circuit, compile_with_d4

def main(ns, inp):
    ids, edges = load_theory(f"dgp2_{ns}"); n = len(ids)
    z = np.load(DATA_DIR / "scores" / f"deepgoplus2_{ns}.npz")
    S, Y, root = z[inp].astype(float), z["labels"].astype(np.int8), int(z["root_index"])
    cols = np.array([j for j in range(n) if j != root]); keep = Y[:, cols].sum(1) > 0
    S, Y = S[keep], Y[keep]
    with tempfile.TemporaryDirectory() as td:
        write_dimacs(Path(td) / "t.cnf", n, edges); compile_with_d4(Path(td) / "t.cnf", Path(td) / "t.nnf")
        c = Circuit(Path(td) / "t.nnf", n); E = np.stack([c.marginals(s)[0] for s in S])
    children = [[] for _ in range(n)]
    for a, b in edges: children[b].append(int(a))
    depth = np.full(n, -1); depth[root] = 0; dq = deque([root])
    while dq:
        u = dq.popleft()
        for v in children[u]:
            if depth[v] < 0: depth[v] = depth[u] + 1; dq.append(v)
    rows = []
    for d in range(1, depth.max() + 1):
        m = depth == d
        if m.sum() == 0: continue
        s, e, y = S[:, m], E[:, m], Y[:, m]
        rows.append(dict(depth=d, terms=int(m.sum()), label_rate=float(y.mean()), raw_mean=float(s.mean()),
                         exact_mean=float(e.mean()), raw_mean_on_pos=float(s[y == 1].mean()) if (y == 1).any() else None,
                         exact_mean_on_pos=float(e[y == 1].mean()) if (y == 1).any() else None,
                         positives=int(y.sum())))
    res = dict(theory=f"dgp2_{ns}", input=inp, proteins=int(len(S)), by_depth=rows)
    json.dump(res, open(RESULTS_DIR / f"p4_2_mechanism_{ns}_{inp}.json", "w"), indent=1)
    print(f"\n{ns} {inp}: depth | terms | label rate | raw mean | exact mean | raw on positives | exact on positives")
    for r in rows:
        print(f"  {r['depth']:2d} {r['terms']:4d}  {r['label_rate']:.4f}  {r['raw_mean']:.4f}  {r['exact_mean']:.4f}   "
              f"{(r['raw_mean_on_pos'] or 0):.3f}  {(r['exact_mean_on_pos'] or 0):.3f}  (pos {r['positives']})")

if __name__ == "__main__":
    for ns in ("go_cc", "go_mf"):
        for inp in ("cnn", "fused"):
            main(ns, inp)
