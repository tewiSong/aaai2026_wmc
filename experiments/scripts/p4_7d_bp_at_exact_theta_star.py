"""P4.7d  Given the exact moment-matched parameters theta* (P4.5 v2 cache), does loopy BP reproduce
the exact marginals? If not, BP cannot serve C2's training or prediction even when theta* is exact.
For 100 random proteins per GO cc / GO mf (CNN input): max and mean |mu_BP(theta*) - mu_exact(theta*)|,
BP convergence, and downstream metrics of BP(theta*) vs exact(theta*) (= the isotonic target)."""
import json, tempfile
from pathlib import Path
import numpy as np
from learned_constrained import log_partition
from p4_3_learned_constrained_model_vs_calibration import metrics
from wmc_common import DATA_DIR, RESULTS_DIR, load_theory, write_dimacs
from wmc_inference import Circuit, compile_with_d4, loopy_bp_theta

for ns in ("go_cc", "go_mf"):
    ids, edges = load_theory(f"dgp2_{ns}"); n = len(ids)
    z = np.load(DATA_DIR / "scores" / f"deepgoplus2_{ns}.npz"); Y, root = z["labels"].astype(np.int8), int(z["root_index"])
    cols = np.array([j for j in range(n) if j != root]); keep = Y[:, cols].sum(1) > 0
    Y = Y[keep]
    c = np.load(DATA_DIR / "scores" / f"theta_star_{ns}_cnn_eps0.001_v2.npz"); TH, err = c["theta"], c["err"]
    have = np.flatnonzero(~np.isnan(err))
    rows = np.random.default_rng(3).choice(have, 100, replace=False)
    td = tempfile.mkdtemp(); write_dimacs(Path(td) / "t.cnf", n, edges); compile_with_d4(Path(td) / "t.cnf", Path(td) / "t.nnf")
    C = Circuit(Path(td) / "t.nnf", n)
    EX, BP, mx, mae, conv = [], [], [], [], []
    for r in rows:
        ex = log_partition(C, TH[r])[1]; bp = loopy_bp_theta(n, edges, TH[r], tol=1e-12, max_iter=20000)
        EX.append(ex); BP.append(bp["mu"]); d = np.abs(bp["mu"] - ex); mx.append(d.max()); mae.append(d.mean()); conv.append(bp["converged"])
    EX, BP = np.array(EX), np.array(BP); Yc = Y[rows][:, cols]
    rec = dict(theory=f"dgp2_{ns}", proteins=len(rows), bp_converged_frac=float(np.mean(conv)),
               max_err_quantiles=np.quantile(mx, [.5, .9, 1]).tolist(), mae_median=float(np.median(mae)),
               frac_max_err_gt_0_1=float(np.mean(np.array(mx) > 0.1)), theta_star_abs_max_median=float(np.median(np.abs(TH[rows]).max(1))),
               metrics=dict(exact_at_theta_star=metrics(EX[:, cols], Yc), bp_at_theta_star=metrics(BP[:, cols], Yc)))
    print(json.dumps(rec), flush=True)
    json.dump(rec, open(RESULTS_DIR / f"p4_7d_bp_at_exact_theta_star_{ns}_cnn.json", "w"), indent=1)
