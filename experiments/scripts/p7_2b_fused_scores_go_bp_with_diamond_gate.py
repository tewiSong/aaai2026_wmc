"""P7.2b  Fused scores (official DeepGOPlus formula) for the P7 GO bp targets, behind gate (b).

Same procedure as P5.2b, for GO bp:
1. DIAMOND v2.2.6 blastp --more-sensitive of the OLD test sequences against train_data.dmnd; old
   GO bp fused scores from our hits vs from the release's test_diamond.res (also checked against
   P4.1a's deepgoplus2_go_bp.npz). Gate (b) passes if the median abs difference <= 1e-3 and the
   old-set raw Fmax changes by < 0.005; if it fails, only the CNN input is confirmed.
2. Same DIAMOND search for the new targets; fused = alpha_bp * DiamondScore + (1 - alpha_bp) * CNN.
3. CNN heads -> vocabulary with the P4.1a rule (alias / replaced_by mapping, max over merged heads).
Outputs: data/scores/p7_scores_go_bp.npz (cnn, fused, labels, accessions aligned with P7.1),
         results/p7_2b_gate_b.json
"""
from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from p4_0_downstream_conditioning_deepgoplus_cc_mf import fmax
from p4_1a_build_deepgoplus_v2_inputs import ALPHA_KEY, go_id_map
from p5_2b_fused_scores_with_diamond_gate import DGP, diamond_scores, run_diamond
from wmc_common import DATA_DIR, RESULTS_DIR, load_theory

NS = "go_bp"


def main():
    idmap = go_id_map()
    terms = pd.read_pickle(DGP / "data" / "terms.pkl")["terms"].tolist()
    mapped = [idmap.get(t, t) for t in terms]
    train = pd.read_pickle(DGP / "data" / "train_data.pkl")
    test = pd.read_pickle(DGP / "data" / "predictions.pkl")
    alpha = json.load(open(DGP / "metadata" / "last_release.json"))["alphas"][ALPHA_KEY[NS]]
    train_annots = {p: {idmap.get(g, g) for g in a} for p, a in zip(train["proteins"], train["prop_annotations"])}
    rel_hits = defaultdict(dict)
    for line in open(DGP / "data" / "test_diamond.res"):
        q, s, b = line.split()
        rel_hits[q][s] = float(b)

    td = Path(tempfile.mkdtemp())
    with open(td / "old_test.fasta", "w") as fh:
        for p, s in zip(test["proteins"], test["sequences"]):
            fh.write(f">{p}\n{s}\n")
    our_old = run_diamond(td / "old_test.fasta", td / "old_test.res")

    ids, edges = load_theory(f"dgp2_{NS}")
    n = len(ids)
    full_ids = load_theory(NS)[0]
    index = {t: i for i, t in enumerate(full_ids)}
    heads = defaultdict(list)
    for k, t in enumerate(mapped):
        if t in index:
            heads[index[t]].append(k)
    vidx = sorted(heads)
    assert [full_ids[g] for g in vidx] == list(ids), "vocabulary differs from P4.1a"
    pos = {g: j for j, g in enumerate(vidx)}

    old = np.load(DATA_DIR / "scores" / f"deepgoplus2_{NS}.npz")
    cnn_old = np.stack([np.stack(test["preds"].values)[:, heads[g]].max(1) for g in vidx], 1)
    fused_rel = alpha * diamond_scores(rel_hits, list(test["proteins"]), train_annots, index, pos, n) + (1 - alpha) * cnn_old
    fused_our = alpha * diamond_scores(our_old, list(test["proteins"]), train_annots, index, pos, n) + (1 - alpha) * cnn_old
    assert np.allclose(np.clip(fused_rel, 1e-6, 1 - 1e-6), old["fused"], atol=1e-5), "release fused differs from P4.1a"
    Y, root = old["labels"].astype(np.int8), int(old["root_index"])
    cols = np.array([j for j in range(n) if j != root]); keep = Y[:, cols].sum(1) > 0
    f_rel = fmax(np.clip(fused_rel, 1e-6, 1 - 1e-6)[keep][:, cols], Y[keep][:, cols])
    f_our = fmax(np.clip(fused_our, 1e-6, 1 - 1e-6)[keep][:, cols], Y[keep][:, cols])
    med = float(np.median(np.abs(fused_rel - fused_our)))
    ok = med <= 1e-3 and abs(f_rel - f_our) < 0.005
    gate = dict(alpha=alpha, fused_abs_diff_median=med, fused_abs_diff_max=float(np.abs(fused_rel - fused_our).max()),
                old_raw_fmax_release=f_rel, old_raw_fmax_recomputed=f_our, passed=bool(ok))

    tz = np.load(DATA_DIR / "scores" / "p7_cnn_heads_go_bp_targets.npz")
    assert list(tz["terms"]) == terms
    tg = np.load(DATA_DIR / "scores" / f"p7_targets_{NS}.npz")
    assert list(tz["accessions"]) == list(tg["accessions"]) and list(tg["vocab_ids"]) == list(ids)
    our_new = run_diamond(DATA_DIR / "scores" / f"p7_targets_{NS}.fasta", td / "targets.res")
    t_heads = tz["scores"].astype(float)
    cnn_new = np.stack([t_heads[:, heads[g]].max(1) for g in vidx], 1)
    fused_new = alpha * diamond_scores(our_new, list(tg["accessions"]), train_annots, index, pos, n) + (1 - alpha) * cnn_new
    np.savez_compressed(DATA_DIR / "scores" / f"p7_scores_{NS}.npz", accessions=tg["accessions"],
                        cnn=np.clip(cnn_new, 1e-6, 1 - 1e-6).astype(np.float32),
                        fused=np.clip(fused_new, 1e-6, 1 - 1e-6).astype(np.float32),
                        labels=tg["labels"], root_index=int(tg["root_index"]), fused_valid=bool(ok))
    gate["targets"] = int(len(tg["accessions"]))
    gate["targets_with_hits"] = int(sum(1 for a in tg["accessions"] if a in our_new))
    print(json.dumps(gate, indent=1))
    json.dump(gate, open(RESULTS_DIR / "p7_2b_gate_b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
