"""P5.2a  DeepGOPlus 1.0.6 CNN scores for the P5 targets, behind reproducibility gate (a).

Runs inside env/tf/deepgoplus_tf.sif (TensorFlow 2.3.1, numpy 1.18.5; deepgoplus commit 585d9c56d3
encoding and chunking, vendored in tools/src/deepgoplus_585d9c56d3).
Gate (a) of the locked protocol: recomputed CNN scores for the 3,874 old test proteins must match
the release's predictions.pkl to max abs difference <= 1e-4; otherwise the script stops before
scoring any new target.
Outputs: data/scores/p5_cnn_heads_targets.npz (accessions x 5,862 heads),
         results/p5_2a_gate_a.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "tools" / "src" / "deepgoplus_585d9c56d3"))
from aminoacids import MAXLEN, to_onehot  # noqa: E402

import tensorflow as tf  # noqa: E402

DGP = EXP / "data" / "raw" / "deepgoplus_1.0.6" / "data"


def get_data(sequences):
    """Identical to deepgoplus main.get_data at 585d9c56d3 (chunks of MAXLEN, stride MAXLEN-128)."""
    pred_seqs, ids = [], []
    for i, seq in enumerate(sequences):
        if len(seq) > MAXLEN:
            st = 0
            while st < len(seq):
                pred_seqs.append(seq[st: st + MAXLEN]); ids.append(i)
                st += MAXLEN - 128
        else:
            pred_seqs.append(seq); ids.append(i)
    data = np.zeros((len(pred_seqs), MAXLEN, 21), dtype=np.float32)
    for i, s in enumerate(pred_seqs):
        data[i] = to_onehot(s)
    return ids, data


def get_data_truncate(sequences):
    """Truncation to MAXLEN (to_onehot on the full sequence). This is how the release's
    predictions.pkl was produced (P5 protocol amendment 1); used for old and new sequences."""
    data = np.zeros((len(sequences), MAXLEN, 21), dtype=np.float32)
    for i, s in enumerate(sequences):
        data[i] = to_onehot(s)
    return list(range(len(sequences))), data


def score(model, sequences, batch=256, long_mode="truncate"):
    out = np.zeros((len(sequences), model.output_shape[-1]), dtype=np.float32)
    for s in range(0, len(sequences), batch):
        ids, data = (get_data_truncate if long_mode == "truncate" else get_data)(sequences[s: s + batch])
        preds = model.predict(data, batch_size=32)
        block = np.zeros((min(batch, len(sequences) - s), preds.shape[1]), dtype=np.float32)
        for row, i in enumerate(ids):
            block[i] = np.maximum(block[i], preds[row])     # main.py keeps the max over chunks
        out[s: s + len(block)] = block
        print(f"scored {s + len(block)}/{len(sequences)}", flush=True)
    return out


def main():
    print("tensorflow", tf.__version__, "numpy", np.__version__, flush=True)
    model = tf.keras.models.load_model(str(DGP / "model.h5"), compile=False)
    test = pd.read_pickle(DGP / "predictions.pkl")
    ref = np.stack(test["preds"].values).astype(np.float32)
    rec = score(model, list(test["sequences"]))
    diff = np.abs(rec - ref)
    gate = dict(proteins=int(len(test)), max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
                passed=bool(diff.max() <= 1e-4), tensorflow=tf.__version__)
    print(json.dumps(gate), flush=True)
    json.dump(gate, open(EXP / "results" / "p5_2a_gate_a.json", "w"), indent=1)
    if not gate["passed"]:
        sys.exit("gate (a) failed: CNN scores do not reproduce the release; stopping before new targets")
    accs, seqs = [], []
    with open(EXP / "data" / "scores" / "p5_targets.fasta") as fh:
        for line in fh:
            if line.startswith(">"):
                accs.append(line[1:].strip()); seqs.append("")
            else:
                seqs[-1] += line.strip()
    S = score(model, seqs)
    np.savez_compressed(EXP / "data" / "scores" / "p5_cnn_heads_targets.npz", accessions=np.array(accs), scores=S,
                        terms=np.array(pd.read_pickle(DGP / "terms.pkl")["terms"].tolist()))
    print(f"scored {len(accs)} targets", flush=True)


if __name__ == "__main__":
    main()
