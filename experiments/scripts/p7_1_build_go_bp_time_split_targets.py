"""P7.1  GO bp time-split targets for the locked GO bp confirmation (P7), same rules as P5.1.

GO bp was never evaluated on the time split (P5 covered GO cc / GO mf only), so it is the clean
confirmation set for the models chosen after P5 (P6.4). Rules are those of the locked P5 protocol
§3–§4, reused from p5_1 unchanged: >= 1 experimental bp annotation in Swiss-Prot 2024_01, no
experimental GO annotation in any namespace in 2021_04 under any accession, no accession / name
overlap and no identical sequence with DeepGOPlus 1.0.6 training or test data; labels propagated in
GO 2024-01-17 (is_a + all relationships), mapped to GO 2021-11-16 ids, restricted to the dgp2_go_bp
vocabulary and closed upward in the dgp2 theory.
Outputs: data/scores/p7_targets_go_bp.npz, data/scores/p7_targets_go_bp.fasta,
         results/p7_1_go_bp_targets_summary.json
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from p5_1_build_time_split_targets import EXP, NSCODE, obo, parse_dat
from wmc_common import DATA_DIR, RAW_DIR, RESULTS_DIR, load_theory

NS, NSNAME, ROOT = "go_bp", "biological_process", "GO:0008150"


def main():
    dgp = RAW_DIR / "deepgoplus_1.0.6" / "data"
    old_ids, old_names, old_seqs = set(), set(), set()
    for df in (pd.read_pickle(dgp / "train_data.pkl"), pd.read_pickle(dgp / "predictions.pkl")):
        for a in df["accessions"]:
            old_ids |= {x.strip() for x in a.split(";") if x.strip()}
        old_names |= set(df["proteins"]); old_seqs |= set(df["sequences"])

    had_exp_2021 = set()
    for accs, name, gos, seq in parse_dat(RAW_DIR / "uniprot_2021_04" / "uniprot_sprot.dat.gz"):
        if any(code in EXP for _, _, code in gos):
            had_exp_2021 |= set(accs)

    par24, alt24, _ = obo(RAW_DIR / "go-basic_2024-01-17.obo", with_rels=True)
    _, alt21, _ = obo(RAW_DIR / "go-basic_2021-11-16.obo", with_rels=False)

    def ancestors(t):
        out, st = set(), [t]
        while st:
            u = st.pop()
            if u not in out:
                out.add(u); st.extend(par24.get(u, ()))
        return out

    ids, edges = load_theory(f"dgp2_{NS}")
    ids = list(ids)
    index = {t: i for i, t in enumerate(ids)}
    parents = [[] for _ in range(len(ids))]
    for c, q in edges:
        parents[c].append(int(q))

    c, targets = defaultdict(int), []
    for accs, name, gos, seq in parse_dat(RAW_DIR / "uniprot_2024_01" / "uniprot_sprot.dat.gz"):
        exp = [g for g, aspect, code in gos if code in EXP and NSCODE.get(aspect) == NSNAME]
        if not exp:
            continue
        c["new_exp_in_ns"] += 1
        if set(accs) & had_exp_2021:
            c["excluded_had_exp_2021"] += 1; continue
        if set(accs) & old_ids or name in old_names:
            c["excluded_id_overlap_deepgoplus"] += 1; continue
        if seq in old_seqs:
            c["excluded_identical_sequence"] += 1; continue
        prop = set()
        for g in exp:
            prop |= ancestors(alt24.get(g, g))
        lab = {index[alt21.get(g, g)] for g in prop if alt21.get(g, g) in index}
        stack = list(lab)
        while stack:
            u = stack.pop()
            for p in parents[u]:
                if p not in lab:
                    lab.add(p); stack.append(p)
        targets.append((accs[0], name, seq, sorted(lab)))

    root = index[ROOT]
    keep = [t for t in targets if any(i != root for i in t[3])]
    c["excluded_no_nonroot_label_in_vocab"] = len(targets) - len(keep)
    Y = np.zeros((len(keep), len(ids)), dtype=np.int8)
    for r, t in enumerate(keep):
        Y[r, t[3]] = 1
    np.savez_compressed(DATA_DIR / "scores" / f"p7_targets_{NS}.npz", accessions=np.array([t[0] for t in keep]),
                        names=np.array([t[1] for t in keep]), labels=Y, vocab_ids=np.array(ids), root_index=root)
    with open(DATA_DIR / "scores" / f"p7_targets_{NS}.fasta", "w") as fh:
        for t in keep:
            fh.write(f">{t[0]}\n{t[2]}\n")
    p5_fasta = {line[1:].strip() for line in open(DATA_DIR / "scores" / "p5_targets.fasta") if line.startswith(">")}
    summary = dict(**c, targets=len(keep), labels_per_target_median=float(np.median(Y.sum(1))) if keep else 0.0,
                   labels_per_target_q90=float(np.quantile(Y.sum(1), 0.9)) if keep else 0.0,
                   also_p5_cc_or_mf_target=int(sum(t[0] in p5_fasta for t in keep)),
                   terms_with_positive=int((Y.sum(0) > 0).sum()), terms=len(ids))
    print(json.dumps(summary), flush=True)
    json.dump(summary, open(RESULTS_DIR / "p7_1_go_bp_targets_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
