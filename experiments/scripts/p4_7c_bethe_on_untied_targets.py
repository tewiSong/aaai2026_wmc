"""P4.7c  Does Bethe moment matching fail only because isotonic targets contain ties?
Targets without ties: exact conditioned marginals of the raw scores, mu* = mu_exact(logit s)
(strictly interior; the true parameters theta = logit s are known). For 30 GO mf proteins:
closed-form Bethe parameters theta_B, BP at theta_B (does it recover mu*?), exact marginals at
theta_B, and |theta_B - logit s| (how far the Bethe parameters are from the true ones)."""
import json, tempfile
from pathlib import Path
import numpy as np
from scipy.special import logit
from learned_constrained import log_partition, moment_match_bethe
from wmc_common import DATA_DIR, RESULTS_DIR, load_theory, write_dimacs
from wmc_inference import Circuit, compile_with_d4, loopy_bp_theta

for ns in ("go_mf", "go_cc"):
    ids, edges = load_theory(f"dgp2_{ns}"); n = len(ids)
    z = np.load(DATA_DIR / "scores" / f"deepgoplus2_{ns}.npz"); S, Y, root = z["cnn"].astype(float), z["labels"], int(z["root_index"])
    keep = np.flatnonzero(np.delete(Y, root, 1).sum(1) > 0)
    rows = np.random.default_rng(2).choice(keep, 30, replace=False)
    td = tempfile.mkdtemp(); write_dimacs(Path(td) / "t.cnf", n, edges); compile_with_d4(Path(td) / "t.cnf", Path(td) / "t.nnf")
    C = Circuit(Path(td) / "t.nnf", n)
    be, xe, de, mg, skipped = [], [], [], [], 0
    for r in rows:
        th_true = logit(np.clip(S[r], 1e-6, 1 - 1e-6))
        mt = log_partition(C, th_true)[1]
        gap = mt[edges[:, 1]] - mt[edges[:, 0]]
        if np.any(gap <= 0):
            skipped += 1; continue
        thb = moment_match_bethe(n, edges, mt, gap=gap)
        bp = loopy_bp_theta(n, edges, thb, tol=1e-12, max_iter=20000)
        be.append(np.abs(bp["mu"] - mt).max()); xe.append(np.abs(log_partition(C, thb)[1] - mt).max())
        de.append(np.abs(thb - th_true).max()); mg.append(gap.min())
    rec = dict(theory=f"dgp2_{ns}", proteins=len(be), skipped_float_ties=skipped,
               bp_recovers_frac=float(np.mean(np.array(be) < 1e-3)), bp_err_median=float(np.median(be)),
               exact_at_bethe_err_median=float(np.median(xe)), theta_bethe_minus_true_median_max=float(np.median(de)),
               min_gap_median=float(np.median(mg)))
    print(json.dumps(rec), flush=True)
    json.dump(rec, open(RESULTS_DIR / f"p4_7c_bethe_untied_{ns}_cnn.json", "w"), indent=1)
