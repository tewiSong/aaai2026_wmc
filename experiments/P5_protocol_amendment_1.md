# P5 protocol amendment 1 (2026-10-22, before any new-target scoring or evaluation)

Context: gate (a) failed on the first P5.2a run (max abs diff 0.805, mean 4e-5). The diagnostic
`scripts/p5_2a_diagnose_gate_a_failure.py` (`results/p5_2a_gate_a_diagnostic.json`) found:
- all 54 old test proteins longer than MAXLEN = 2000 differ when scored with the chunk-and-max rule
  of deepgoplus main.py (max diff 0.805);
- with plain truncation to MAXLEN (`to_onehot` on the full sequence), the release scores are
  reproduced to max abs diff 3.3e-7, for long and short sequences alike.

Amendment: the release's predictions.pkl, and therefore the fitting data of all methods, was
produced by **truncation**. To keep old and new scores consistent, P5 scores every sequence,
old and new, by truncation to MAXLEN. Gate (a) is unchanged (≤ 1e-4) and is re-applied.

No new-target score had been computed, and no evaluation had been run, when this amendment was
made. The locked protocol file is unchanged; this amendment is recorded separately.
