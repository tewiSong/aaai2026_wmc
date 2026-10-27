"""P4.1b  Standardised downstream comparison of consistency methods on DeepGOPlus 1.0.6.

Purpose: decide whether any WMC-based post-processing has a role for marginal correction,
against the established WMC-free baseline (isotonic regression = closest consistent marginals,
Obozinski et al. 2008), after fixing the evaluation issues found in the Codex P4.0 audit.

Inputs  : cnn (release CNN branch) and fused (official alpha * DiamondScore + (1-alpha) * cnn)
          from P4.1a, vocabulary theories dgp2_<ns>.
Methods : raw | max_closure | isotonic (L2; identical to Bernoulli-KL(p||mu) projection, see
          self-test) | exact_wmc (d4 circuit; skipped when the theory is not compiled) | bp
Protocol: namespace root excluded from all metrics; proteins with >= 1 non-root label;
          protein-centric Fmax (0.01 grid), micro average precision over (protein, term) pairs
          with ties handled at tie-group ends ("micro_ap", not the official AUPR), Brier,
          pooled 15-bin ECE, hierarchy-violation rate; 1,000 paired bootstrap resamples of
          proteins for differences vs raw and vs isotonic.

Output: results/p4_1b_downstream_<ns>_<input>.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from consistency import isotonic_dag_hildreth
from p4_0_downstream_conditioning_deepgoplus_cc_mf import ece, fmax, max_closure
from wmc_common import DATA_DIR, RESULTS_DIR, load_theory, write_dimacs
from wmc_inference import Circuit, compile_with_d4, loopy_bp


def micro_ap(P, Y):
    s, y = P.ravel(), Y.ravel()
    o = np.argsort(-s, kind="stable")
    s, y = s[o], y[o]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]     # last index of each tie group
    tp_e, fp_e = tp[ends], fp[ends]
    prec = tp_e / (tp_e + fp_e)
    rec = tp_e / max(y.sum(), 1)
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def metrics(P, Y, edges):
    return dict(fmax=fmax(P, Y), micro_ap=micro_ap(P, Y), brier=float(np.mean((P - Y) ** 2)),
                ece=ece(P, Y), violation_rate=float(np.mean(P[:, edges[:, 0]] > P[:, edges[:, 1]] + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True)
    ap.add_argument("--input", choices=["cnn", "fused"], required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--exact", choices=["auto", "off"], default="auto")
    args = ap.parse_args()

    name = f"dgp2_{args.ns}"
    ids, edges = load_theory(name)
    n = len(ids)
    z = np.load(DATA_DIR / "scores" / f"deepgoplus2_{args.ns}.npz")
    S, Y, root = z[args.input].astype(np.float64), z["labels"].astype(np.int8), int(z["root_index"])
    cols = np.array([j for j in range(n) if j != root])
    keep = Y[:, cols].sum(1) > 0
    S, Y = S[keep], Y[keep]
    timing, outs = {}, {"raw": S}
    t0 = time.time(); outs["max_closure"] = max_closure(S, edges, n); timing["max_closure"] = time.time() - t0
    t0 = time.time(); iso, info = isotonic_dag_hildreth(S, edges); timing["isotonic"] = time.time() - t0
    outs["isotonic"] = iso
    if args.exact == "auto" and n <= 1000:
        with tempfile.TemporaryDirectory() as td:
            cnf, nnf = Path(td) / "t.cnf", Path(td) / "t.nnf"
            write_dimacs(cnf, n, edges)
            t0 = time.time(); compile_with_d4(cnf, nnf); circ = Circuit(nnf, n)
            outs["exact_wmc"] = np.stack([circ.marginals(s)[0] for s in S]); timing["exact_wmc"] = time.time() - t0
    t0 = time.time(); outs["bp"] = np.stack([loopy_bp(n, edges, s)["mu"] for s in S]); timing["bp"] = time.time() - t0

    ev = {k: v[:, cols] for k, v in outs.items()}
    Yc = Y[:, cols]
    # edges among non-root columns, for the violation rate
    remap = -np.ones(n, dtype=int); remap[cols] = np.arange(len(cols))
    e2 = remap[edges]; e2 = e2[(e2 >= 0).all(1)]
    res = dict(theory=name, input=args.input, terms_evaluated=int(len(cols)), proteins=int(len(S)),
               isotonic_solver=info, seconds_total=timing,
               metrics={k: metrics(v, Yc, e2) for k, v in ev.items()})
    rng = np.random.default_rng(0)
    refs = ["raw", "isotonic"]
    diffs = {r: {k: {"fmax": [], "micro_ap": []} for k in ev if k != r} for r in refs}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(S), len(S))
        m = {k: (fmax(v[idx], Yc[idx]), micro_ap(v[idx], Yc[idx])) for k, v in ev.items()}
        for r in refs:
            for k in diffs[r]:
                diffs[r][k]["fmax"].append(m[k][0] - m[r][0])
                diffs[r][k]["micro_ap"].append(m[k][1] - m[r][1])
    res["method_minus_ref_95ci"] = {r: {k: {mm: np.quantile(v, [0.025, 0.5, 0.975]).tolist() for mm, v in d.items()}
                                        for k, d in dd.items()} for r, dd in diffs.items()}
    print(json.dumps({k: res[k] for k in ("theory", "input", "proteins", "metrics")}, indent=1))
    json.dump(res, open(RESULTS_DIR / f"p4_1b_downstream_{args.ns}_{args.input}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
