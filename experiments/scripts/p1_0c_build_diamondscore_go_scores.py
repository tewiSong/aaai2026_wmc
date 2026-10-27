"""P1.0c  Realistic predictor scores: DiamondScore homology transfer on Swiss-Prot 2021_04.

Purpose: P1.0a showed (smoke test) that BP is badly wrong under uniform/Beta scores but
near-exact under a synthetic sparse proxy. Whether "high-accuracy WMC" matters therefore
hinges on what real predictor scores look like. DiamondScore is the standard CAFA homology
baseline: score_t(q) = sum_s bitscore(q,s) [t in A(s)] / sum_s bitscore(q,s) over DIAMOND
hits s of query q in the annotated training set.

Two variants per GO namespace
  prop : training annotations propagated to all ancestors (scores respect the hierarchy)
  raw  : unpropagated experimental annotations (scores can violate the hierarchy, as
         independent per-term classifiers such as DeepGO-style models do)
Proteins with experimental GO annotations are split 80/20 (seed 0) into train/test.

Output: data/scores/diamondscore_<go_ns>.npz (CSR matrices test-proteins x terms for
prop and raw, true propagated labels, accessions); results/p1_0c_diamondscore_summary.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from wmc_common import DATA_DIR, EXP_DIR, RAW_DIR, RESULTS_DIR, load_theory

EXPERIMENTAL = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC",
                "HTP", "HDA", "HMP", "HGI", "HEP"}
NAMESPACES = ["go_cc", "go_mf", "go_bp"]
UNI = RAW_DIR / "uniprot_2021_04"
DIAMOND = EXP_DIR / "tools" / "bin" / "diamond"


def go_alt_ids():
    alt, cur = {}, None
    with open(RAW_DIR / "go-basic_2021-11-16.obo") as fh:
        for line in fh:
            if line.startswith("id: "):
                cur = line.split()[1]
            elif line.startswith("alt_id: ") and cur:
                alt[line.split()[1]] = cur
            elif line.startswith("["):
                cur = None
    return alt


def parse_swissprot():
    annots, acc = defaultdict(set), None
    with gzip.open(UNI / "uniprot_sprot.dat.gz", "rt") as fh:
        for line in fh:
            if line.startswith("AC   ") and acc is None:
                acc = line[5:].split(";")[0].strip()
            elif line.startswith("DR   GO;"):
                f = [x.strip() for x in line[5:].split(";")]
                code = f[3].split(":")[0]
                if code in EXPERIMENTAL:
                    annots[acc].add(f[1])
            elif line.startswith("//"):
                acc = None
    seqs, cur, buf = {}, None, []
    with gzip.open(UNI / "uniprot_sprot.fasta.gz", "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur:
                    seqs[cur] = "".join(buf)
                cur, buf = line.split("|")[1], []
            else:
                buf.append(line.strip())
    if cur:
        seqs[cur] = "".join(buf)
    return annots, seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--max-proteins", type=int, default=0, help="smoke tests only")
    ap.add_argument("--out-dir", default=str(DATA_DIR / "scores"))
    args = ap.parse_args()
    work = Path(args.out_dir)
    work.mkdir(parents=True, exist_ok=True)

    annots, seqs = parse_swissprot()
    alt = go_alt_ids()
    theories = {}
    for ns in NAMESPACES:
        ids, edges = load_theory(ns)
        index = {t: i for i, t in enumerate(ids)}
        parents = [[] for _ in range(len(ids))]
        for c, q in edges:
            parents[c].append(int(q))
        theories[ns] = (ids, index, parents)

    def closure(terms, parents):
        out, stack = set(), list(terms)
        while stack:
            u = stack.pop()
            if u not in out:
                out.add(u)
                stack.extend(parents[u])
        return out

    raw = {ns: {} for ns in NAMESPACES}
    unmapped = 0
    for acc, gos in annots.items():
        if acc not in seqs:
            continue
        for g in gos:
            g = alt.get(g, g)
            hit = False
            for ns in NAMESPACES:
                i = theories[ns][1].get(g)
                if i is not None:
                    raw[ns].setdefault(acc, set()).add(i)
                    hit = True
            unmapped += not hit
    proteins = sorted(set().union(*[set(raw[ns]) for ns in NAMESPACES]))
    rng = np.random.default_rng(0)
    if args.max_proteins:
        proteins = sorted(rng.choice(proteins, args.max_proteins, replace=False).tolist())
    test = set(np.array(proteins)[rng.random(len(proteins)) < 0.2].tolist())
    train = [a for a in proteins if a not in test]
    test = sorted(test)

    for name, accs in [("train", train), ("test", test)]:
        with open(work / f"{name}.fasta", "w") as fh:
            for a in accs:
                fh.write(f">{a}\n{seqs[a]}\n")
    subprocess.run([str(DIAMOND), "makedb", "--in", str(work / "train.fasta"),
                    "-d", str(work / "train"), "--threads", str(args.threads)], check=True)
    hits_tsv = work / "test_vs_train.tsv"
    subprocess.run([str(DIAMOND), "blastp", "-d", str(work / "train.dmnd"), "-q", str(work / "test.fasta"),
                    "-o", str(hits_tsv), "--more-sensitive", "-e", "0.001", "-k", "100",
                    "--outfmt", "6", "qseqid", "sseqid", "bitscore", "--threads", str(args.threads)],
                   check=True)
    hits = defaultdict(list)
    with open(hits_tsv) as fh:
        for line in fh:
            q, s, b = line.split()
            hits[q].append((s, float(b)))

    summary = dict(proteins_annotated=len(proteins), train=len(train), test=len(test),
                   annotations_unmapped=unmapped, test_with_hits=len(hits))
    for ns in NAMESPACES:
        ids, index, parents = theories[ns]
        n = len(ids)
        prop_train = {a: closure(t, parents) for a, t in raw[ns].items()}
        rows = {"prop": ([], [], []), "raw": ([], [], [])}
        labels_idx, labels_ptr, accs = [], [0], []
        for q in test:
            hs = [(s, b) for s, b in hits.get(q, []) if s in raw[ns]]
            if not hs or q not in raw[ns]:
                continue
            r = len(accs)
            accs.append(q)
            tot = sum(b for _, b in hs)
            for key, src in (("prop", prop_train), ("raw", raw[ns])):
                acc_s = defaultdict(float)
                for s, b in hs:
                    for t in src[s]:
                        acc_s[t] += b / tot
                for t, v in acc_s.items():
                    rows[key][0].append(r)
                    rows[key][1].append(t)
                    rows[key][2].append(v)
            lab = sorted(closure(raw[ns][q], parents))
            labels_idx.extend(lab)
            labels_ptr.append(len(labels_idx))
        save = dict(accessions=np.array(accs), n_terms=n,
                    labels_indices=np.array(labels_idx, dtype=np.int32),
                    labels_indptr=np.array(labels_ptr, dtype=np.int64))
        for key, (r, c, v) in rows.items():
            m = csr_matrix((np.array(v, dtype=np.float32), (r, c)), shape=(len(accs), n))
            save[f"{key}_data"], save[f"{key}_indices"], save[f"{key}_indptr"] = m.data, m.indices, m.indptr
            nnz = np.diff(m.indptr)
            summary[f"{ns}_{key}_nonzero_terms_per_protein_median"] = float(np.median(nnz)) if len(nnz) else 0
        np.savez_compressed(work / f"diamondscore_{ns}.npz", **save)
        summary[f"{ns}_test_proteins"] = len(accs)
        print(ns, len(accs), "test proteins", flush=True)

    summary_path = (RESULTS_DIR / "p1_0c_diamondscore_summary.json" if not args.max_proteins
                    else work / "smoke_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
