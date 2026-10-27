"""Independent P4.0 audit. Run in the existing project container.

Reads existing scores/circuits without changing experiment artifacts. Recomputes
reported point estimates, tests root exclusion and tied-score AP, and checks a
few circuit marginals with independent dense variable elimination. Bootstrap
uses paired protein resamples and precomputed threshold sufficient statistics.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
from wmc_common import load_theory, undirected_adjacency
from wmc_inference import Circuit, ve_log_z
from p0_3_minfill_width_check_vs_paper import eliminate_greedy
from p4_0_downstream_conditioning_deepgoplus_cc_mf import max_closure, fmax, aupr, ece


def grouped_ap(P, Y):
    s, y = P.ravel(), Y.ravel()
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tp = np.cumsum(y)[ends]
    return float(np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / max(y.sum(), 1))


def point_metrics(P, Y):
    return dict(fmax=float(fmax(P, Y)), sequential_ap=float(aupr(P, Y)),
                grouped_micro_ap=grouped_ap(P, Y),
                brier=float(np.mean((P - Y) ** 2)), ece=float(ece(P, Y)))


def threshold_stats(P, Y):
    prec, rec, has = [], [], []
    truth = Y.sum(1)
    for t in np.linspace(0.01, 0.99, 99):
        pred = P >= t
        nn = pred.sum(1)
        tp = (pred & Y.astype(bool)).sum(1)
        prec.append(tp / np.maximum(nn, 1))
        rec.append(tp / np.maximum(truth, 1))
        has.append(nn > 0)
    return np.array(prec).T, np.array(rec).T, np.array(has, dtype=float).T


def boot_fmax(P, Y, counts):
    prec, rec, has = threshold_stats(P, Y)
    pr = (counts @ prec) / np.maximum(counts @ has, 1)
    re = (counts @ rec) / counts.sum(1, keepdims=True)
    return (2 * pr * re / np.maximum(pr + re, 1e-300)).max(1)


def ca_fa_like_macro_aupr(P, Y):
    # Same macro precision/recall averaging as Fmax, distinct from micro AP.
    prec, rec, has = threshold_stats(P, Y)
    pr = prec.sum(0) / np.maximum(has.sum(0), 1)
    re = rec.mean(0)
    idx = np.argsort(re)
    return float(np.trapz(pr[idx], re[idx]))


def exact_fmax_score_sweep(P, Y):
    """Protein-centric Fmax over all distinct scores, grouping score ties."""
    n, terms = P.shape
    order = np.argsort(-P.ravel(), kind="stable")
    scores = P.ravel()[order]
    y = Y.ravel()[order]
    rows = order // terms
    by_row = np.argsort(rows, kind="stable")
    yr, rr = y[by_row], rows[by_row]
    starts = np.r_[0, np.flatnonzero(rr[1:] != rr[:-1]) + 1]
    lengths = np.diff(np.r_[starts, len(rr)])
    count = np.arange(len(rr)) - np.repeat(starts, lengths) + 1
    cumulative = np.cumsum(yr)
    before_group = np.r_[0, cumulative[starts[1:] - 1]]
    tp = cumulative - np.repeat(before_group, lengths)
    dp = tp / count - (tp - yr) / np.maximum(count - 1, 1)
    dh = count == 1
    dr = yr / np.maximum(Y.sum(1)[rr], 1)
    inv = np.empty_like(by_row)
    inv[by_row] = np.arange(len(by_row))
    sum_prec = np.cumsum(dp[inv])
    has = np.cumsum(dh[inv])
    sum_rec = np.cumsum(dr[inv])
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    pr, re = sum_prec[ends] / has[ends], sum_rec[ends] / n
    f = 2 * pr * re / np.maximum(pr + re, 1e-300)
    best = int(np.argmax(f))
    return dict(fmax=float(f[best]), threshold=float(scores[ends[best]]))


def main():
    result = dict(author="OpenAI Codex", date="2026-10-20", namespaces={})
    for ns, root_id in [("go_cc", "GO:0005575"), ("go_mf", "GO:0003674")]:
        z = np.load(ROOT / f"experiments/data/scores/deepgoplus_{ns}.npz")
        ids, edges = load_theory("dgp_" + ns)
        S, Y = z["scores"].astype(float), z["labels"].astype(np.int8)
        keep = Y.sum(1) > 0
        S, Y, accs = S[keep], Y[keep], z["accessions"][keep]
        circuit = Circuit(ROOT / f"experiments/data/compiled/dgp_{ns}.nnf", len(ids))
        E = np.stack([circuit.marginals(p)[0] for p in S])
        M = max_closure(S, edges, len(ids))
        original = {k: point_metrics(p, Y) for k, p in
                    [("raw", S), ("max_closure", M), ("exact_wmc", E)]}
        saved = json.load(open(ROOT / f"experiments/results/p4_0_downstream_dgp_{ns}.json"))
        reproduction_error = {
            k: {m: abs(original[k][m] - saved["metrics"][k][m])
                for m in ["fmax", "brier", "ece"]} for k in original}
        reproduction_error["aupr"] = {
            k: abs(original[k]["sequential_ap"] - saved["metrics"][k]["aupr"])
            for k in original}

        order_info = eliminate_greedy(undirected_adjacency(len(ids), edges), "min-fill")
        order = order_info["order"]
        checks = []
        # Include labels that were strongly suppressed, not only arbitrary easy terms.
        suppression = np.where(Y == 1, S - E, -np.inf)
        ri, ti = np.unravel_index(np.argmax(suppression), suppression.shape)
        for row, term in [(0, 0), (len(S) // 2, len(ids) // 2), (int(ri), int(ti))]:
            mu, lz = circuit.marginals(S[row])
            ve_lz = ve_log_z(len(ids), edges, S[row], order)
            ve_mu = np.exp(ve_log_z(len(ids), edges, S[row], order,
                                  clamp={term: 1}) - ve_lz)
            checks.append(dict(protein=str(accs[row]), term=str(ids[term]),
                               circuit=float(mu[term]), ve=float(ve_mu),
                               abs_error=float(abs(mu[term] - ve_mu)),
                               logz_difference=float(abs(lz - ve_lz))))

        nonroot = ids != root_id
        yr = Y[:, nonroot]
        positive = yr.sum(1) > 0
        yr = yr[positive]
        outs = {"raw": S[positive][:, nonroot],
                "max_closure": M[positive][:, nonroot],
                "exact_wmc": E[positive][:, nonroot]}
        root_excluded = {k: point_metrics(p, yr) for k, p in outs.items()}
        for k, p in outs.items():
            root_excluded[k]["macro_pr_auc_99_thresholds"] = ca_fa_like_macro_aupr(p, yr)
        rng = np.random.default_rng(0)
        counts = np.zeros((1000, len(yr)))
        for b in range(1000):
            idx = rng.integers(0, len(yr), len(yr))
            counts[b] = np.bincount(idx, minlength=len(yr))
        boot = {k: boot_fmax(p, yr, counts) for k, p in outs.items()}
        ci = {k: np.quantile(boot["exact_wmc"] - v, [.025, .5, .975]).tolist()
              for k, v in boot.items() if k != "exact_wmc"}

        # Counts alone do not establish a causal mechanism; retain denominators.
        shift = E - S
        mechanism = dict(positive_total=int(Y.sum()), negative_total=int((Y == 0).sum()),
                         positive_drops_over_0_1=int(((Y == 1) & (shift < -.1)).sum()),
                         negative_rises_over_0_1=int(((Y == 0) & (shift > .1)).sum()),
                         mean_positive_shift=float(shift[Y == 1].mean()),
                         mean_negative_shift=float(shift[Y == 0].mean()))
        result["namespaces"][ns] = dict(
            terms=len(ids), proteins_original=len(Y),
            original_definition_recomputed=original,
            max_absolute_reproduction_errors=reproduction_error,
            independent_ve_checks=checks, independent_ve_width=order_info["width"],
            root_excluded_positive_proteins=len(yr), root_only_proteins=int((~positive).sum()),
            root_excluded_metrics=root_excluded,
            root_excluded_paired_fmax_bootstrap_95ci=ci,
            mechanism_counts=mechanism)
        print(ns, json.dumps(result["namespaces"][ns], ensure_ascii=False), flush=True)
    sources = ["p4_0_downstream_conditioning_deepgoplus_cc_mf.py",
               "p1_4_build_deepgoplus_vocab_theories_and_scores.py", "wmc_inference.py"]
    result["source_hashes"] = {f: hashlib.sha256(
        (ROOT / "experiments/scripts" / f).read_bytes()).hexdigest() for f in sources}
    target = ROOT / "journal_comparison/Codex_P4.0_audit_2026-10-20.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("Saved", str(target), flush=True)


if __name__ == "__main__":
    main()
