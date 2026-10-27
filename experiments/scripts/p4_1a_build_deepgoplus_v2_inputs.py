"""P4.1a  Standardised DeepGOPlus 1.0.6 inputs for the downstream study (fixes from the Codex
P4.0 audit, 2026-10-20).

Changes w.r.t. P1.4 (whose files are kept unchanged for reproducibility):
  * vocabulary mapping: GO alt_id -> primary id and obsolete -> single replaced_by; terms that
    still do not map are dropped (listed). If several predictor heads map to the same current
    term, the term's score is the MAX over those heads (rule fixed before any evaluation).
  * two score inputs: 'cnn' (release CNN branch) and 'fused' = alpha*DiamondScore +
    (1-alpha)*cnn exactly as in the official evaluate_deepgoplus.py (hits from
    test_diamond.res, training prop_annotations, alphas from metadata/last_release.json).
  * labels: release prop_annotations mapped to current ids and closed upwards in GO 2021-11-16
    (removes the GO bp child-1/parent-0 label pairs), restricted to the vocabulary.
  * namespace root ids saved so evaluation can exclude them (official protocol).

Outputs: data/theories/dgp2_<ns>.npz/.cnf; data/scores/deepgoplus2_<ns>.npz;
         results/p4_1a_inputs_summary.json
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from p1_4_build_deepgoplus_vocab_theories_and_scores import project
from wmc_common import DATA_DIR, RAW_DIR, RESULTS_DIR, THEORY_DIR, load_theory, save_theory, write_dimacs

DGP = RAW_DIR / "deepgoplus_1.0.6"
ROOTS = {"go_cc": "GO:0005575", "go_mf": "GO:0003674", "go_bp": "GO:0008150"}
ALPHA_KEY = {"go_cc": "cc", "go_mf": "mf", "go_bp": "bp"}


def go_id_map():
    """alt_id -> id, obsolete id -> replaced_by (only when unique), from GO 2021-11-16."""
    m, cur, obs, repl = {}, None, False, []
    def flush():
        if cur and obs and len(repl) == 1:
            m[cur] = repl[0]
    with open(RAW_DIR / "go-basic_2021-11-16.obo") as fh:
        for line in fh:
            if line.startswith("["):
                flush(); cur, obs, repl = None, False, []
            elif line.startswith("id: "):
                cur = line.split()[1]
            elif line.startswith("alt_id: ") and cur:
                m[line.split()[1]] = cur
            elif line.startswith("is_obsolete: true"):
                obs = True
            elif line.startswith("replaced_by: "):
                repl.append(line.split()[1])
    flush()
    return m


def main():
    idmap = go_id_map()
    terms = pd.read_pickle(DGP / "data" / "terms.pkl")["terms"].tolist()
    preds = pd.read_pickle(DGP / "data" / "predictions.pkl")
    train = pd.read_pickle(DGP / "data" / "train_data.pkl")
    alphas = json.load(open(DGP / "metadata" / "last_release.json"))["alphas"]
    cnn_all = np.stack(preds["preds"].values).astype(np.float64)
    test_ids = preds["proteins"].tolist()
    mapped_terms = [idmap.get(t, t) for t in terms]

    hits = defaultdict(dict)
    with open(DGP / "data" / "test_diamond.res") as fh:
        for line in fh:
            q, s, b = line.split()
            hits[q][s] = float(b)
    train_annots = {p: {idmap.get(g, g) for g in a} for p, a in zip(train["proteins"], train["prop_annotations"])}

    summary = dict(test_proteins=len(test_ids), heads=len(terms), alphas=alphas)
    for ns in ["go_cc", "go_mf", "go_bp"]:
        ids, edges = load_theory(ns)
        index = {t: i for i, t in enumerate(ids)}
        parents = [[] for _ in range(len(ids))]
        for c, q in edges:
            parents[c].append(int(q))
        heads_by_term = defaultdict(list)
        for k, t in enumerate(mapped_terms):
            if t in index:
                heads_by_term[index[t]].append(k)
        vidx = sorted(heads_by_term)
        pos = {g: j for j, g in enumerate(vidx)}
        n = len(vidx)
        sub = np.array(sorted((pos[c], pos[q]) for c, q in project(ids, edges, vidx)), dtype=np.int32).reshape(-1, 2)
        name = f"dgp2_{ns}"
        save_theory(name, [ids[g] for g in vidx], sub)
        write_dimacs(THEORY_DIR / f"{name}.cnf", n, sub, comment=f"DeepGOPlus v2 vocabulary of {ns}")

        cnn = np.stack([cnn_all[:, heads_by_term[g]].max(1) for g in vidx], axis=1)
        alpha = alphas[ALPHA_KEY[ns]]
        dia = np.zeros_like(cnn)
        for i, prot in enumerate(test_ids):
            h = hits.get(prot)
            if not h:
                continue
            tot = sum(h.values())
            for s, b in h.items():
                for g in train_annots.get(s, ()):
                    j = pos.get(index.get(g, -1))
                    if j is not None:
                        dia[i, j] += b / tot
        fused = alpha * dia + (1 - alpha) * cnn

        labels = np.zeros((len(test_ids), n), dtype=np.int8)
        for i, annots in enumerate(preds["prop_annotations"]):
            stack = [index[idmap.get(g, g)] for g in annots if idmap.get(g, g) in index]
            seen = set()
            while stack:  # upward closure in the current theory
                u = stack.pop()
                if u not in seen:
                    seen.add(u)
                    stack.extend(parents[u])
            for u in seen:
                if u in pos:
                    labels[i, pos[u]] = 1
        viol = int(np.sum(labels[:, sub[:, 0]] > labels[:, sub[:, 1]])) if len(sub) else 0
        root = pos.get(index[ROOTS[ns]])
        np.savez_compressed(DATA_DIR / "scores" / f"deepgoplus2_{ns}.npz",
                            cnn=np.clip(cnn, 1e-6, 1 - 1e-6).astype(np.float32),
                            fused=np.clip(fused, 1e-6, 1 - 1e-6).astype(np.float32),
                            labels=labels, proteins=np.array(test_ids), root_index=-1 if root is None else root,
                            vocab_ids=np.array([ids[g] for g in vidx]))
        summary[ns] = dict(vocabulary_terms=n, projected_edges=int(len(sub)),
                           heads_merged=int(sum(len(v) > 1 for v in heads_by_term.values())),
                           heads_via_alt_or_replaced=int(sum(1 for t, mt in zip(terms, mapped_terms) if t != mt and mt in index)),
                           root_in_vocab=root is not None, label_violations_after_closure=viol,
                           proteins_with_labels_excl_root=int(((labels.sum(1) - (labels[:, root] if root is not None else 0)) > 0).sum()),
                           alpha=alpha, fused_differs_from_cnn=float(np.mean(np.abs(fused - cnn) > 1e-9)))
        print(ns, json.dumps(summary[ns]), flush=True)
    summary["heads_unmapped"] = [t for t, mt in zip(terms, mapped_terms)
                                 if not any(mt in set(load_theory(ns)[0].tolist()) for ns in ROOTS)]
    print("unmapped heads:", summary["heads_unmapped"])
    json.dump(summary, open(RESULTS_DIR / "p4_1a_inputs_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
