"""Additional audit of CNN+DIAMOND scores on the SAME retained vocabulary.

This does not reproduce the official full-label DeepGOPlus benchmark: its
DIAMOND component also predicts labels outside the CNN vocabulary. The check
isolates whether the P4 negative result extends to the release's hybrid scores
under the exact same ontology and vocabulary-restricted evaluation task.
"""
from pathlib import Path
import io
import json
import pickle
import sys
import tarfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
from wmc_common import load_theory
from wmc_inference import Circuit
from p4_0_downstream_conditioning_deepgoplus_cc_mf import max_closure, fmax
from p4_0_audit_2026_10_20 import point_metrics, exact_fmax_score_sweep


def main():
    # Check the all-score sweep against a literal exhaustive threshold scan.
    tiny = np.array([[.1, .1, .8], [.5, .2, .2]])
    lab = np.array([[1, 0, 1], [0, 1, 0]])
    assert abs(exact_fmax_score_sweep(tiny, lab)["fmax"] -
               fmax(tiny, lab, thresholds=np.unique(tiny))) < 1e-12
    base = ROOT / "experiments/data/raw/deepgoplus_1.0.6"
    cache = Path("/tmp/wmc_p4_codex_hybrid_inputs_20261020.pkl")
    if cache.exists():
        with cache.open("rb") as f:
            train, hits = pickle.load(f)
    else:
        train, hits = None, None
        with tarfile.open(base / "data-1.0.6.tar.gz", "r|gz") as archive:
            for member in archive:
                if member.name == "data/train_data.pkl":
                    train = pd.read_pickle(io.BytesIO(archive.extractfile(member).read()))
                elif member.name == "data/test_diamond.res":
                    hits = archive.extractfile(member).read().decode()
                if train is not None and hits is not None:
                    break
        if train is None or hits is None:
            raise RuntimeError("Training annotations or DIAMOND scores absent from release archive")
    pred = pd.read_pickle(base / "data/predictions.pkl")
    terms = pd.read_pickle(base / "data/terms.pkl")["terms"].tolist()
    term_index = {t: k for k, t in enumerate(terms)}
    train_ann = {str(r.proteins): set(r.prop_annotations) for r in train.itertuples()}
    by_query = {}
    for line in hits.splitlines():
        q, subject, bitscore = line.split()
        by_query.setdefault(q, {})[subject] = float(bitscore)
    alpha = json.load(open(base / "metadata/last_release.json"))["alphas"]
    output = dict(date="2026-10-20", scope=__doc__, namespaces={})
    raw_cnn = np.stack(pred.preds.values).astype(float)
    for ns, root_id in [("go_cc", "GO:0005575"), ("go_mf", "GO:0003674")]:
        z = np.load(ROOT / f"experiments/data/scores/deepgoplus_{ns}.npz")
        ids, edges = load_theory("dgp_" + ns)
        ix = [term_index[t] for t in ids]
        pos = {t: i for i, t in enumerate(ids)}
        D = np.zeros((len(pred), len(ids)))
        for row, q in enumerate(pred.proteins):
            hs = by_query.get(q, {})
            total = sum(hs.values())
            for subject, bitscore in hs.items():
                for term in train_ann[subject]:
                    if term in pos:
                        D[row, pos[term]] += bitscore / total
        a = alpha[ns[-2:]]
        S = np.clip(a * D + (1 - a) * raw_cnn[:, ix], 1e-6, 1-1e-6)
        Y = z["labels"]
        nonroot = ids != root_id
        keep = Y[:, nonroot].sum(1) > 0
        S, Y = S[keep], Y[keep]
        circuit = Circuit(ROOT / f"experiments/data/compiled/dgp_{ns}.nnf", len(ids))
        E = np.stack([circuit.marginals(p)[0] for p in S])
        M = max_closure(S, edges, len(ids))
        yr = Y[:, nonroot]
        outs = {"hybrid_raw": S[:, nonroot], "hybrid_max_closure": M[:, nonroot],
                "hybrid_exact_wmc": E[:, nonroot]}
        ms = {k: point_metrics(p, yr) for k, p in outs.items()}
        for k, p in outs.items():
            ms[k]["all_distinct_score_fmax"] = exact_fmax_score_sweep(p, yr)
        # Also verify that the CNN negative result is not a 0.01 threshold-grid artifact.
        C = z["scores"].astype(float)[keep]
        CE = np.stack([circuit.marginals(p)[0] for p in C])
        CM = max_closure(C, edges, len(ids))
        cnn_sweep = {k: exact_fmax_score_sweep(p[:, nonroot], yr)
                     for k, p in [("raw", C), ("max_closure", CM), ("exact_wmc", CE)]}
        out = dict(proteins=len(Y), terms_evaluated=int(nonroot.sum()), alpha=a,
                   hybrid_root_excluded_metrics=ms, cnn_all_score_fmax=cnn_sweep)
        output["namespaces"][ns] = out
        print(ns, json.dumps(out), flush=True)
    path = ROOT / "journal_comparison/Codex_P4.0_hybrid_audit_2026-10-20.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print("Saved", str(path), flush=True)


if __name__ == "__main__":
    main()
