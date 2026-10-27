"""P4.7a  Can BP / Bethe reproduce C2's starting point? Diagnostic on the loopy GO vocabulary
theories. Target mu* = 0.999 * isotonic(s) + 0.001 * mu_ref, mu_ref = exact marginals at p = 0.1
(strictly inside the order polytope, away from 1 to avoid floating-point ties near the root).
For each protein: closed-form Bethe parameters theta_B (moment_match_bethe); BP from the
uniform initialisation at theta_B; exact marginals at theta_B; exact moment matching theta*.
Reports how often BP recovers mu*, how far exact marginals at theta_B are from mu*, and the
metrics of the three starting points (isotonic target, BP(theta_B), exact(theta_B)).
Output: results/p4_7a_bethe_diagnostic_<ns>_<input>.json
"""
import argparse, json, tempfile, time
from pathlib import Path
import numpy as np
from consistency import enforce_order_exact, isotonic_dag_hildreth
from learned_constrained import log_partition, moment_match_bethe
from p4_3_learned_constrained_model_vs_calibration import metrics
from wmc_common import DATA_DIR, RESULTS_DIR, load_theory, write_dimacs
from wmc_inference import Circuit, compile_with_d4, loopy_bp_theta

ap = argparse.ArgumentParser(); ap.add_argument("--ns", default="go_mf"); ap.add_argument("--input", default="cnn")
ap.add_argument("--proteins", type=int, default=200); args = ap.parse_args()
ids, edges = load_theory(f"dgp2_{args.ns}"); n = len(ids)
z = np.load(DATA_DIR / "scores" / f"deepgoplus2_{args.ns}.npz")
S, Y, root = z[args.input].astype(float), z["labels"].astype(np.int8), int(z["root_index"])
cols = np.array([j for j in range(n) if j != root]); keep = np.flatnonzero(Y[:, cols].sum(1) > 0)
rows = np.random.default_rng(0).choice(keep, min(args.proteins, len(keep)), replace=False)
td = tempfile.mkdtemp(); write_dimacs(Path(td) / "t.cnf", n, edges); compile_with_d4(Path(td) / "t.cnf", Path(td) / "t.nnf")
C = Circuit(Path(td) / "t.nnf", n)
mu_ref, _ = C.marginals(np.full(n, 0.1))
assert np.all(mu_ref[edges[:, 1]] > mu_ref[edges[:, 0]])
T, B, X, rec = [], [], [], []
t0 = time.time()
for r in rows:
    s = np.clip(S[r], 1e-6, 1 - 1e-6)
    iso = enforce_order_exact(isotonic_dag_hildreth(s[None], edges)[0], edges)[0]
    mt = 0.999 * iso + 0.001 * mu_ref
    gap = 0.999 * (iso[edges[:, 1]] - iso[edges[:, 0]]) + 0.001 * (mu_ref[edges[:, 1]] - mu_ref[edges[:, 0]])
    thb = moment_match_bethe(n, edges, mt, gap=gap)
    bp = loopy_bp_theta(n, edges, thb, tol=1e-12, max_iter=20000)
    ex = log_partition(C, thb)[1]
    T.append(mt); B.append(bp["mu"]); X.append(ex)
    rec.append(dict(bp_converged=bp["converged"], bp_err=float(np.abs(bp["mu"] - mt).max()),
                    exact_at_bethe_err=float(np.abs(ex - mt).max()), exact_at_bethe_mae=float(np.abs(ex - mt).mean())))
T, B, X = map(np.array, (T, B, X)); Yc = Y[rows][:, cols]
be, xe = np.array([q["bp_err"] for q in rec]), np.array([q["exact_at_bethe_err"] for q in rec])
out = dict(theory=f"dgp2_{args.ns}", input=args.input, proteins=len(rows), seconds=round(time.time() - t0, 1),
           bp_converged_frac=float(np.mean([q["bp_converged"] for q in rec])),
           bp_recovers_target_frac_1e3=float(np.mean(be < 1e-3)), bp_err_quantiles=np.quantile(be, [.5, .9, 1]).tolist(),
           exact_at_bethe_err_quantiles=np.quantile(xe, [.5, .9, 1]).tolist(),
           exact_at_bethe_mae_median=float(np.median([q["exact_at_bethe_mae"] for q in rec])),
           metrics={k: metrics(v[:, cols], Yc) for k, v in dict(isotonic_target=T, bp_at_bethe=B, exact_at_bethe=X).items()})
print(json.dumps(out, indent=1))
json.dump(out, open(RESULTS_DIR / f"p4_7a_bethe_diagnostic_{args.ns}_{args.input}.json", "w"), indent=1)
