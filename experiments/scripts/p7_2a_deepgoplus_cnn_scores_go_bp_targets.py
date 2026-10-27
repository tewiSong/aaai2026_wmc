"""P7.2a  DeepGOPlus 1.0.6 CNN head scores for the P7 GO bp targets, behind the P5 gate (a).

Same container (env/tf/deepgoplus_tf.sif), model and encoding as P5.2a (truncation to MAXLEN,
P5 protocol amendment 1). Gate (a) is re-run first: recomputed CNN scores for the 3,874 old test
proteins must match the release's predictions.pkl to max abs difference <= 1e-4.
Outputs: data/scores/p7_cnn_heads_go_bp_targets.npz, results/p7_2a_gate_a.json
"""
import json
import sys

import numpy as np
import pandas as pd

from p5_2a_deepgoplus_cnn_scores_with_reproducibility_gate import DGP, EXP, score, tf


def main():
    print("tensorflow", tf.__version__, "numpy", np.__version__, flush=True)
    model = tf.keras.models.load_model(str(DGP / "model.h5"), compile=False)
    test = pd.read_pickle(DGP / "predictions.pkl")
    ref = np.stack(test["preds"].values).astype(np.float32)
    diff = np.abs(score(model, list(test["sequences"])) - ref)
    gate = dict(proteins=int(len(test)), max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
                passed=bool(diff.max() <= 1e-4), tensorflow=tf.__version__)
    print(json.dumps(gate), flush=True)
    json.dump(gate, open(EXP / "results" / "p7_2a_gate_a.json", "w"), indent=1)
    if not gate["passed"]:
        sys.exit("gate (a) failed: CNN scores do not reproduce the release; stopping before new targets")
    accs, seqs = [], []
    with open(EXP / "data" / "scores" / "p7_targets_go_bp.fasta") as fh:
        for line in fh:
            if line.startswith(">"):
                accs.append(line[1:].strip()); seqs.append("")
            else:
                seqs[-1] += line.strip()
    S = score(model, seqs)
    np.savez_compressed(EXP / "data" / "scores" / "p7_cnn_heads_go_bp_targets.npz", accessions=np.array(accs), scores=S,
                        terms=np.array(pd.read_pickle(DGP / "terms.pkl")["terms"].tolist()))
    print(f"scored {len(accs)} targets", flush=True)


if __name__ == "__main__":
    main()
