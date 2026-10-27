"""P5.1  Build the time-split target set (protocol P5_protocol_time_split_confirmation.md, §3–§4).

Compares Swiss-Prot 2021_04 with 2024_01. A target in namespace ns has >= 1 experimental ns
annotation in 2024_01, no experimental GO annotation in ANY namespace in 2021_04 under any of its
accessions, no accession / name overlap with DeepGOPlus 1.0.6 training or test data, and no exact
sequence match with those data. New labels are propagated in GO 2024-01-17 over is_a + all
relationships (DeepGOPlus with_rels=True), mapped to GO 2021-11-16 ids, restricted to the dgp2
vocabulary and closed upward in the dgp2 theory.

Outputs: data/scores/p5_targets_<ns>.npz, data/scores/p5_targets.fasta,
         results/p5_1_targets_summary.json
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from wmc_common import DATA_DIR, RAW_DIR, RESULTS_DIR, load_theory

EXP = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC", "HTP", "HDA", "HMP", "HGI", "HEP"}
NS = {"go_cc": "cellular_component", "go_mf": "molecular_function"}
NSCODE = {"C": "cellular_component", "F": "molecular_function", "P": "biological_process"}


def parse_dat(path):
    """Yield (accessions, name, [(go, aspect, code)], sequence) per Swiss-Prot entry."""
    accs, name, gos, seq, in_seq = [], None, [], [], False
    with gzip.open(path, "rt") as fh:
        for line in fh:
            tag = line[:2]
            if tag == "ID":
                name = line[5:].split()[0]
            elif tag == "AC":
                accs += [a.strip() for a in line[5:].split(";") if a.strip()]
            elif line.startswith("DR   GO;"):
                f = [x.strip() for x in line[5:].split(";")]
                gos.append((f[1], f[2][:1], f[3].split(":")[0]))
            elif tag == "SQ":
                in_seq = True
            elif tag == "//":
                yield accs, name, gos, "".join(seq)
                accs, name, gos, seq, in_seq = [], None, [], [], False
            elif in_seq:
                seq.append(line.strip().replace(" ", ""))


def obo(path, with_rels):
    """id -> parents (is_a [+ all relationships]), alt_id -> id, obsolete -> unique replaced_by, namespace."""
    par, alt, ns = defaultdict(list), {}, {}
    cur, obs, repl = None, False, []
    def flush():
        if cur and obs and len(repl) == 1:
            alt[cur] = repl[0]
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("["):
                flush(); cur, obs, repl = None, False, []
            elif line.startswith("id: "):
                cur = line.split()[1]
            elif line.startswith("alt_id: ") and cur:
                alt[line.split()[1]] = cur
            elif line.startswith("namespace: ") and cur:
                ns[cur] = line.split()[1]
            elif line.startswith("is_a: ") and cur:
                par[cur].append(line.split()[1])
            elif with_rels and line.startswith("relationship: ") and cur:
                par[cur].append(line.split()[2])
            elif line.startswith("is_obsolete: true"):
                obs = True
            elif line.startswith("replaced_by: "):
                repl.append(line.split()[1])
    flush()
    return par, alt, ns


def main():
    dgp = RAW_DIR / "deepgoplus_1.0.6" / "data"
    train = pd.read_pickle(dgp / "train_data.pkl")
    test = pd.read_pickle(dgp / "predictions.pkl")
    old_ids, old_names, old_seqs = set(), set(), set()
    for df in (train, test):
        for a in df["accessions"]:
            old_ids |= {x.strip() for x in a.split(";") if x.strip()}
        old_names |= set(df["proteins"]); old_seqs |= set(df["sequences"])

    had_exp_2021 = set()
    for accs, name, gos, seq in parse_dat(RAW_DIR / "uniprot_2021_04" / "uniprot_sprot.dat.gz"):
        if any(code in EXP for _, _, code in gos):
            had_exp_2021 |= set(accs)

    par24, alt24, ns24 = obo(RAW_DIR / "go-basic_2024-01-17.obo", with_rels=True)
    _, alt21, _ = obo(RAW_DIR / "go-basic_2021-11-16.obo", with_rels=False)

    def ancestors(t):
        out, st = set(), [t]
        while st:
            u = st.pop()
            if u not in out:
                out.add(u); st.extend(par24.get(u, ()))
        return out

    theories = {}
    for ns in NS:
        ids, edges = load_theory(f"dgp2_{ns}")
        parents = [[] for _ in range(len(ids))]
        for c, q in edges:
            parents[c].append(int(q))
        theories[ns] = (list(ids), {t: i for i, t in enumerate(ids)}, parents)

    counts = defaultdict(lambda: defaultdict(int))
    targets = {ns: [] for ns in NS}
    for accs, name, gos, seq in parse_dat(RAW_DIR / "uniprot_2024_01" / "uniprot_sprot.dat.gz"):
        for ns, nsname in NS.items():
            exp = [g for g, aspect, code in gos if code in EXP and NSCODE.get(aspect) == nsname]
            if not exp:
                continue
            c = counts[ns]; c["new_exp_in_ns"] += 1
            if set(accs) & had_exp_2021:
                c["excluded_had_exp_2021"] += 1; continue
            if set(accs) & old_ids or name in old_names:
                c["excluded_id_overlap_deepgoplus"] += 1; continue
            if seq in old_seqs:
                c["excluded_identical_sequence"] += 1; continue
            ids, index, parents = theories[ns]
            prop = set()
            for g in exp:
                prop |= ancestors(alt24.get(g, g))
            lab = set()
            for g in prop:
                g21 = alt21.get(g, g)
                if g21 in index:
                    lab.add(index[g21])
            stack = list(lab)
            while stack:
                u = stack.pop()
                for p in parents[u]:
                    if p not in lab:
                        lab.add(p); stack.append(p)
            targets[ns].append((accs[0], name, seq, sorted(lab)))

    summary = {}
    fasta = {}
    for ns in NS:
        ids, index, parents = theories[ns]
        root = index["GO:0005575" if ns == "go_cc" else "GO:0003674"]
        keep = [t for t in targets[ns] if any(i != root for i in t[3])]
        counts[ns]["excluded_no_nonroot_label_in_vocab"] = len(targets[ns]) - len(keep)
        Y = np.zeros((len(keep), len(ids)), dtype=np.int8)
        for r, t in enumerate(keep):
            Y[r, t[3]] = 1
            fasta[t[0]] = t[2]
        np.savez_compressed(DATA_DIR / "scores" / f"p5_targets_{ns}.npz", accessions=np.array([t[0] for t in keep]),
                            names=np.array([t[1] for t in keep]), labels=Y, vocab_ids=np.array(ids), root_index=root)
        summary[ns] = dict(**counts[ns], targets=len(keep), labels_per_target_median=float(np.median(Y.sum(1))) if len(keep) else 0)
        print(ns, json.dumps(summary[ns]), flush=True)
    with open(DATA_DIR / "scores" / "p5_targets.fasta", "w") as fh:
        for a, s in fasta.items():
            fh.write(f">{a}\n{s}\n")
    summary["fasta_sequences"] = len(fasta)
    json.dump(summary, open(RESULTS_DIR / "p5_1_targets_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
