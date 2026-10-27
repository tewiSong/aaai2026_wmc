"""Summarise P1.0a JSON results: BP / Gibbs error against exact d4 marginals, per ontology,
instance size and score source (synthetic family or real DiamondScore prop/raw)."""
import glob
import json
from collections import defaultdict

import numpy as np

from wmc_common import RESULTS_DIR

rows = defaultdict(list)
fails = defaultdict(int)
inst_info = defaultdict(list)
for f in sorted(glob.glob(str(RESULTS_DIR / "p1_0a_*.json"))):
    d = json.load(open(f))
    a = d["args"]
    src = ("real-" + a["diamond_key"]) if a.get("diamond") else "synthetic"
    for inst in d["instances"]:
        key0 = (a["ontology"], a["subgraph_size"], src)
        if "runs" not in inst:
            fails[key0] += 1
            continue
        inst_info[key0].append((inst["terms"], inst["pct_multiparent"], inst["d4"]["seconds"], inst["circuit"]["nodes"]))
        for r in inst["runs"]:
            fam = r["scores"].split(":")[0] if src == "synthetic" else src
            rows[(a["ontology"], a["subgraph_size"], fam)].append(r)

print(f"{'ontology':8s} {'size':>5s} {'scores':10s} {'n':>3s} | {'BP MAE med':>10s} {'BP max med':>10s} {'BP max worst':>12s} {'frac>1e-2':>9s} {'viol':>5s} | {'Gibbs max med':>13s} {'brkt>0.05':>9s}")
for key in sorted(rows):
    rs = rows[key]
    bp_mae = np.median([r["bp"]["mae"] for r in rs]); bp_max = [r["bp"]["max"] for r in rs]
    frac = np.median([r["bp"]["frac_gt_1e2"] for r in rs]); viol = sum(r["bp"]["violations"] for r in rs)
    g_max = np.median([r["gibbs"]["max"] for r in rs]); brk = np.mean([r["gibbs"]["max_bracket_gap"] > 0.05 for r in rs])
    print(f"{key[0]:8s} {key[1]:5d} {key[2]:10s} {len(rs):3d} | {bp_mae:10.2e} {np.median(bp_max):10.2e} {max(bp_max):12.2e} {frac:9.3f} {viol:5d} | {g_max:13.2e} {brk:9.2f}")
print("\ninstances (terms, %multiparent, d4 s, circuit nodes) and d4 failures:")
for key in sorted(set(inst_info) | set(fails)):
    info = inst_info.get(key, [])
    print(key, "ok:", [(t, m, round(s, 1), c) for t, m, s, c in info][:4], "failed:", fails.get(key, 0))
# real scores: relation between number of intermediate scores and BP error
print("\nreal scores, BP max error vs number of intermediate scores in instance:")
for src in ("real-prop", "real-raw"):
    pts = [(r["n_intermediate"], r["bp"]["max"]) for k, rs in rows.items() if k[2] == src for r in rs]
    if pts:
        pts = np.array(pts)
        for lo, hi in [(0, 5), (5, 20), (20, 100), (100, 10**6)]:
            m = (pts[:, 0] >= lo) & (pts[:, 0] < hi)
            if m.any():
                print(f"  {src} n_inter in [{lo},{hi}): {m.sum():3d} runs, BP max median {np.median(pts[m,1]):.2e}, worst {pts[m,1].max():.2e}")
