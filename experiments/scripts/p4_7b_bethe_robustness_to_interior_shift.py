"""P4.7b  Is the Bethe / BP failure of P4.7a an artifact of the interior shift?
Sweep eps in {1e-3, 1e-2, 1e-1} and reference p_ref in {0.1, 0.3, 0.5}: target
mu* = (1-eps) iso + eps mu_ref (gaps computed directly). For each setting report the fraction of
proteins where BP at the closed-form Bethe parameters recovers mu* (max err < 1e-3), the median max
error of BP and of exact marginals at those parameters, and the smallest edge gap."""
import json, tempfile
from pathlib import Path
import numpy as np
from consistency import enforce_order_exact, isotonic_dag_hildreth
from learned_constrained import log_partition, moment_match_bethe
from wmc_common import DATA_DIR, RESULTS_DIR, load_theory, write_dimacs
from wmc_inference import Circuit, compile_with_d4, loopy_bp_theta

ns, inp = "go_mf", "cnn"
ids, edges = load_theory(f"dgp2_{ns}"); n = len(ids)
z = np.load(DATA_DIR / "scores" / f"deepgoplus2_{ns}.npz")
S, Y, root = z[inp].astype(float), z["labels"], int(z["root_index"])
keep = np.flatnonzero(np.delete(Y, root, 1).sum(1) > 0)
rows = np.random.default_rng(1).choice(keep, 30, replace=False)
td = tempfile.mkdtemp(); write_dimacs(Path(td) / "t.cnf", n, edges); compile_with_d4(Path(td) / "t.cnf", Path(td) / "t.nnf")
C = Circuit(Path(td) / "t.nnf", n)
ISO = enforce_order_exact(isotonic_dag_hildreth(np.clip(S[rows], 1e-6, 1 - 1e-6), edges)[0], edges)
out = []
for pref in (0.1, 0.3, 0.5):
    ref, _ = C.marginals(np.full(n, pref))
    gref = ref[edges[:, 1]] - ref[edges[:, 0]]
    for eps in (1e-3, 1e-2, 1e-1):
        be, xe, mg = [], [], []
        for iso in ISO:
            mt = (1 - eps) * iso + eps * ref
            gap = (1 - eps) * (iso[edges[:, 1]] - iso[edges[:, 0]]) + eps * gref
            thb = moment_match_bethe(n, edges, mt, gap=gap)
            bp = loopy_bp_theta(n, edges, thb, tol=1e-12, max_iter=20000)
            be.append(np.abs(bp["mu"] - mt).max()); xe.append(np.abs(log_partition(C, thb)[1] - mt).max()); mg.append(gap.min())
        rec = dict(p_ref=pref, eps=eps, bp_recovers_frac=float(np.mean(np.array(be) < 1e-3)),
                   bp_err_median=float(np.median(be)), exact_err_median=float(np.median(xe)), min_gap_median=float(np.median(mg)))
        out.append(rec); print(json.dumps(rec), flush=True)
json.dump(out, open(RESULTS_DIR / "p4_7b_bethe_robustness_go_mf_cnn.json", "w"), indent=1)
