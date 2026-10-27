"""P5.2b  Fused scores (official DeepGOPlus formula) for P5 targets, behind reproducibility gate (b).

1. DIAMOND v2.2.6 blastp --more-sensitive (release defaults) of the OLD test sequences against the
   release's train_data.dmnd; compare with the release's test_diamond.res (pair agreement,
   bitscores). Recompute old fused scores from our hits and compare with the release-based fused
   scores of P4.1a: gate (b) passes if the median abs difference <= 1e-3 and the old-set raw Fmax
   changes by < 0.005 (per namespace). If it fails, only the CNN input is confirmed (protocol §2).
2. Same DIAMOND search for the new targets; DiamondScore from the release's training
   prop_annotations; fused = alpha * DiamondScore + (1 - alpha) * CNN (alpha cc .52, mf .67).
3. CNN heads -> vocabulary with the P4.1a rule (alias / replaced_by mapping, max over merged heads).
Outputs: data/scores/p5_scores_<ns>.npz (cnn, fused, labels, accessions aligned with P5.1),
         results/p5_2b_gate_b.json
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from p4_0_downstream_conditioning_deepgoplus_cc_mf import fmax
from p4_1a_build_deepgoplus_v2_inputs import ALPHA_KEY, go_id_map
from wmc_common import DATA_DIR, EXP_DIR, RAW_DIR, RESULTS_DIR, load_theory

DGP = RAW_DIR / "deepgoplus_1.0.6"
DIAMOND = EXP_DIR / "tools" / "bin" / "diamond"


def run_diamond(fasta: Path, out: Path):
    subprocess.run([str(DIAMOND), "blastp", "-d", str(DGP / "data" / "train_data.dmnd"), "--more-sensitive",
                    "-q", str(fasta), "-o", str(out), "--outfmt", "6", "qseqid", "sseqid", "bitscore",
                    "--threads", "16"], check=True)
    hits = defaultdict(dict)
    for line in open(out):
        q, s, b = line.split()
        hits[q][s] = float(b)
    return hits


def diamond_scores(hits, query_ids, train_annots, index, pos, n):
    D = np.zeros((len(query_ids), n))
    for i, q in enumerate(query_ids):
        h = hits.get(q)
        if not h:
            continue
        tot = sum(h.values())
        for s, b in h.items():
            for g in train_annots.get(s, ()):
                j = pos.get(index.get(g, -1))
                if j is not None:
                    D[i, j] += b / tot
    return D


def main():
    idmap = go_id_map()
    terms = pd.read_pickle(DGP / "data" / "terms.pkl")["terms"].tolist()
    mapped = [idmap.get(t, t) for t in terms]
    train = pd.read_pickle(DGP / "data" / "train_data.pkl")
    test = pd.read_pickle(DGP / "data" / "predictions.pkl")
    alphas = json.load(open(DGP / "metadata" / "last_release.json"))["alphas"]
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
    rel_pairs = {(q, s) for q, d in rel_hits.items() for s in d}
    our_pairs = {(q, s) for q, d in our_old.items() for s in d}
    common = rel_pairs & our_pairs
    bdiff = [abs(rel_hits[q][s] - our_old[q][s]) for q, s in common]
    gate = dict(release_pairs=len(rel_pairs), our_pairs=len(our_pairs), common_pairs=len(common),
                jaccard=len(common) / max(len(rel_pairs | our_pairs), 1),
                bitscore_abs_diff_median=float(np.median(bdiff)) if bdiff else None)

    tz = np.load(DATA_DIR / "scores" / "p5_cnn_heads_targets.npz")
    t_accs, t_cnn_heads = list(tz["accessions"]), tz["scores"].astype(float)
    assert list(tz["terms"]) == terms
    with open(td / "targets.fasta", "w") as fh:
        for line in open(DATA_DIR / "scores" / "p5_targets.fasta"):
            fh.write(line)
    our_new = run_diamond(td / "targets.fasta", td / "targets.res")

    passed_all = True
    for ns in ("go_cc", "go_mf"):
        ids, edges = load_theory(f"dgp2_{ns}")
        n = len(ids); index = {t: i for i, t in enumerate(load_theory(ns)[0])}
        full_ids = load_theory(ns)[0]
        heads = defaultdict(list)
        for k, t in enumerate(mapped):
            if t in index:
                heads[index[t]].append(k)
        vidx = sorted(heads)
        assert [full_ids[g] for g in vidx] == list(ids), "vocabulary differs from P4.1a"
        pos = {g: j for j, g in enumerate(vidx)}
        alpha = alphas[ALPHA_KEY[ns]]
        old = np.load(DATA_DIR / "scores" / f"deepgoplus2_{ns}.npz")
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
        passed_all &= ok
        gate[ns] = dict(fused_abs_diff_median=med, fused_abs_diff_max=float(np.abs(fused_rel - fused_our).max()),
                        old_raw_fmax_release=f_rel, old_raw_fmax_recomputed=f_our, passed=bool(ok))

        tg = np.load(DATA_DIR / "scores" / f"p5_targets_{ns}.npz")
        rows = [t_accs.index(a) for a in tg["accessions"]]
        cnn_new = np.stack([t_cnn_heads[rows][:, heads[g]].max(1) for g in vidx], 1)
        fused_new = alpha * diamond_scores(our_new, list(tg["accessions"]), train_annots, index, pos, n) + (1 - alpha) * cnn_new
        assert list(tg["vocab_ids"]) == list(ids)
        np.savez_compressed(DATA_DIR / "scores" / f"p5_scores_{ns}.npz", accessions=tg["accessions"],
                            cnn=np.clip(cnn_new, 1e-6, 1 - 1e-6).astype(np.float32),
                            fused=np.clip(fused_new, 1e-6, 1 - 1e-6).astype(np.float32),
                            labels=tg["labels"], root_index=int(tg["root_index"]), fused_valid=bool(ok))
        gate[ns]["targets"] = int(len(rows)); gate[ns]["targets_with_hits"] = int(sum(1 for a in tg["accessions"] if a in our_new))
    gate["passed_all"] = bool(passed_all)
    print(json.dumps(gate, indent=1))
    json.dump(gate, open(RESULTS_DIR / "p5_2b_gate_b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
