"""Summarise P1.2: active-core sizes, d4 success, certificate delta, and BP's certified error
on the full ontology with real DiamondScore vectors."""
import glob
import json

import numpy as np

from wmc_common import RESULTS_DIR

for f in sorted(glob.glob(str(RESULTS_DIR / "p1_2_certified_*.json"))):
    d = json.load(open(f))
    ps = d["proteins"]
    ok = [p for p in ps if "failed" not in p]
    fail = [p for p in ps if "failed" in p]
    name = f.split("/")[-1]
    print(f"\n== {name}: {len(ps)} proteins, terms={d['terms']}, d4 ok {len(ok)}, failed {len(fail)}")
    if fail:
        print("   failed: nonzero-score terms", sorted(p["n_nonzero"] for p in fail)[:20])
    if not ok:
        continue
    core = np.array([p["core_size"] for p in ok]); t = np.array([p["core_seconds"] for p in ok])
    print(f"   core size quartiles {np.quantile(core,[.25,.5,.75]).round(0)}, max {core.max()}; "
          f"core seconds median {np.median(t):.2f}, 90% {np.quantile(t,.9):.1f}, max {t.max():.1f}")
    print(f"   delta max {max(p['delta'] for p in ok):.1e}; max interval width {max(p['max_interval_width'] for p in ok):.1e}")
    bp_s = np.array([p["bp_seconds"] for p in ok])
    print(f"   BP seconds median {np.median(bp_s):.1f}; converged {np.mean([p['bp_converged'] for p in ok]):.2f}")
    lb = np.array([p["bp_certified_error_lower_bound"] for p in ok])
    out = np.array([p["bp_terms_outside_interval"] for p in ok])
    for sel in ("most_intermediate", "random"):
        m = np.array([p["selection"] == sel for p in ok])
        if m.any():
            print(f"   [{sel:17s}] n={m.sum():3d}  BP certified error >1e-3: {np.mean(lb[m]>1e-3):.2f}  >1e-2: {np.mean(lb[m]>1e-2):.2f}  "
                  f"median {np.median(lb[m]):.1e}  max {lb[m].max():.3f}  | terms outside median {np.median(out[m]):.0f}, max {out[m].max()}")
