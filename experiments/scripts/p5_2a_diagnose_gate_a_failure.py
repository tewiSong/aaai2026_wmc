"""P5.2a diagnostic: which old test proteins fail gate (a), and does a different handling of
long sequences (truncation to MAXLEN instead of chunk-max) reproduce the release scores?"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "tools" / "src" / "deepgoplus_585d9c56d3"))
sys.path.insert(0, str(EXP / "scripts"))
from aminoacids import MAXLEN, to_onehot
import tensorflow as tf
from p5_2a_deepgoplus_cnn_scores_with_reproducibility_gate import score
DGP = EXP / "data" / "raw" / "deepgoplus_1.0.6" / "data"
model = tf.keras.models.load_model(str(DGP / "model.h5"), compile=False)
test = pd.read_pickle(DGP / "predictions.pkl")
ref = np.stack(test["preds"].values).astype(np.float32)
lens = test["sequences"].str.len().values
longi = np.flatnonzero(lens > MAXLEN)
rng = np.random.default_rng(0); short = rng.choice(np.flatnonzero(lens <= MAXLEN), 300, replace=False)
sel = np.r_[longi, short]
seqs = [test["sequences"].iloc[i] for i in sel]
chunk = score(model, seqs, long_mode="chunk")
trunc = model.predict(np.stack([to_onehot(s).astype(np.float32) for s in seqs]), batch_size=32)
d_chunk, d_trunc = np.abs(chunk - ref[sel]).max(1), np.abs(trunc - ref[sel]).max(1)
out = dict(n_long=int(len(longi)), n_short_checked=300,
           long_chunk_maxdiff=float(d_chunk[:len(longi)].max()) if len(longi) else None,
           long_trunc_maxdiff=float(d_trunc[:len(longi)].max()) if len(longi) else None,
           short_chunk_maxdiff=float(d_chunk[len(longi):].max()), short_trunc_maxdiff=float(d_trunc[len(longi):].max()),
           frac_long_with_diff_gt_1e4_chunk=float(np.mean(d_chunk[:len(longi)] > 1e-4)) if len(longi) else None)
print(json.dumps(out, indent=1))
json.dump(out, open(EXP / "results" / "p5_2a_gate_a_diagnostic.json", "w"), indent=1)
