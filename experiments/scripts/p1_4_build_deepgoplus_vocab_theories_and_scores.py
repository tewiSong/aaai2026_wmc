"""P1.4  Second, dense real score source: DeepGOPlus 1.0.6 CNN predictions.

Purpose (Codex audit): P1.2's small cores came from DiamondScore's exact zeros. Neural
predictors output a small non-zero score for every term in their vocabulary, which is the
regime that decides whether a dedicated method is needed. DeepGOPlus 1.0.6 (trained on GO
2021-10-26 and Swiss-Prot 2021_04) ships its CNN predictions for 3,874 held-out test proteins
over 5,862 terms.

Scores exist only for the predictor's vocabulary, so we do NOT zero-fill other terms. Each GO
namespace theory is projected onto the vocabulary: y_c => y_a is kept whenever a is reachable
from c in GO 2021-11-16 through non-vocabulary terms only (existential projection of a Horn
implication chain), so no constraint among predicted terms is lost.

Outputs
  data/theories/dgp_<ns>.npz/.cnf      vocabulary theories (ids, edges)
  data/scores/deepgoplus_<ns>.npz      CNN scores (test proteins x vocabulary), labels, accessions
  results/p1_4_deepgoplus_profile.json score-regime profile per namespace
"""
from __future__ import annotations

import json
from collections import deque

import numpy as np
import pandas as pd

from wmc_common import DATA_DIR, RAW_DIR, RESULTS_DIR, THEORY_DIR, load_theory, save_theory, write_dimacs
from wmc_inference import cold_set_for_budget

DGP = RAW_DIR / "deepgoplus_1.0.6" / "data"


def project(ids, edges, vocab_idx):
    """Edges among vocabulary terms after existentially projecting out all other terms."""
    n = len(ids)
    parents = [[] for _ in range(n)]
    for c, q in edges:
        parents[c].append(int(q))
    invocab = np.zeros(n, dtype=bool)
    invocab[vocab_idx] = True
    out = set()
    for v in vocab_idx:
        seen, dq = set(), deque(parents[v])
        while dq:
            u = dq.popleft()
            if u in seen:
                continue
            seen.add(u)
            if invocab[u]:
                out.add((int(v), u))       # nearest vocabulary ancestor on this path
            else:
                dq.extend(parents[u])
    return out


def main():
    terms = pd.read_pickle(DGP / "terms.pkl")["terms"].tolist()
    preds = pd.read_pickle(DGP / "predictions.pkl")
    P = np.stack(preds["preds"].values).astype(np.float64)       # proteins x 5862
    L = np.stack(preds["labels"].values).astype(np.int8)
    accs = np.array([a.split(";")[0].strip() for a in preds["accessions"]])
    profile = dict(test_proteins=int(len(preds)), vocabulary=len(terms))

    for ns in ["go_cc", "go_mf", "go_bp"]:
        ids, edges = load_theory(ns)
        index = {t: i for i, t in enumerate(ids)}
        cols = [k for k, t in enumerate(terms) if t in index]
        vidx = [index[terms[k]] for k in cols]
        pos = {g: j for j, g in enumerate(vidx)}
        proj = project(ids, edges, vidx)
        sub = np.array(sorted((pos[c], pos[q]) for c, q in proj), dtype=np.int32).reshape(-1, 2)
        name = f"dgp_{ns}"
        save_theory(name, [ids[g] for g in vidx], sub)
        write_dimacs(THEORY_DIR / f"{name}.cnf", len(vidx), sub, comment=f"DeepGOPlus vocabulary of {ns}")

        # vocabulary closure in the full theory: how many ancestors of predicted terms are unpredicted?
        parents = [[] for _ in range(len(ids))]
        for c, q in edges:
            parents[c].append(int(q))
        vset, missing = set(vidx), set()
        for v in vidx:
            stack = list(parents[v])
            while stack:
                u = stack.pop()
                if u not in vset and u not in missing:
                    missing.add(u)
                stack.extend(w for w in parents[u] if w not in vset and w not in missing)

        S = np.clip(P[:, cols], 1e-6, 1 - 1e-6)
        sums = S.sum(1)
        inter = ((S > 0.05) & (S < 0.95)).sum(1)
        n = len(vidx)
        pct_mp = 100 * float(np.mean(np.bincount(sub[:, 0], minlength=n) >= 2)) if len(sub) else 0.0
        viol = np.mean([np.mean(S[i, sub[:, 0]] > S[i, sub[:, 1]] + 1e-12) for i in range(min(500, len(S)))])
        cores = {}
        for target in (1e-3, 1e-2, 1e-1):
            sizes = [int((~cold_set_for_budget(n, sub, S[i], target)[0]).sum()) for i in range(min(300, len(S)))]
            cores[f"{target:g}"] = dict(median=float(np.median(sizes)), p90=float(np.quantile(sizes, 0.9)),
                                       max=int(max(sizes)))
        profile[ns] = dict(
            vocabulary_terms=n, projected_edges=int(len(sub)), pct_multiparent=round(pct_mp, 2),
            unpredicted_ancestor_terms=len(missing), vocab_terms_not_in_go2021=len([t for t in terms if t not in index]) if ns == "go_cc" else None,
            score_sum_quantiles=np.quantile(sums, [0.1, 0.5, 0.9]).round(2).tolist(),
            min_score_median=float(np.median(S.min(1))),
            n_intermediate_quantiles=np.quantile(inter, [0.1, 0.5, 0.9]).tolist(),
            frac_edges_violated_by_scores=round(float(viol), 4),
            certified_core_size_for_delta=cores,
        )
        np.savez_compressed(DATA_DIR / "scores" / f"deepgoplus_{ns}.npz", scores=S.astype(np.float32),
                            labels=L[:, cols], accessions=accs, vocab_ids=np.array([ids[g] for g in vidx]))
        print(ns, json.dumps(profile[ns]), flush=True)

    with open(RESULTS_DIR / "p1_4_deepgoplus_profile.json", "w") as fh:
        json.dump(profile, fh, indent=2)


if __name__ == "__main__":
    main()
