# P5 protocol: independent time-split confirmation of C2 (locked before any evaluation)

Written by Claude on 2026-10-22, after the Codex audit `journal_comparison/Codex_P4.7v2判断与独立验证条件_2026-10-22.md`.
The protocol is frozen before any new-snapshot evaluation runs. Its sha256 is recorded in
`results/p5_protocol_lock.txt`, and every P5 evaluation script checks that hash first.
Any later deviation must be recorded as a separate, dated amendment.

## 1. Question

Does the learned constrained model C2 (exact moment-matched θ\* plus a learned residual) beat the
fair non-WMC methods on proteins whose GO annotations appeared only after the predictor and the
fitting data were frozen?

## 2. Frozen information (old side)

- **Predictor**: DeepGOPlus release 1.0.6 (`data-1.0.6.tar.gz`, sha256 8c63ad16…), trained on GO
  2021-10-26 and Swiss-Prot 2021_04.
  - CNN branch: `model.h5`, run with the pinned code (deepgoplus commit 585d9c56d3,
    TensorFlow 2.3.1, MAXLEN = 2000).
  - Fused score: α·DiamondScore + (1−α)·CNN, with α from `metadata/last_release.json` (cc 0.52,
    mf 0.67). DIAMOND searches against the release's `train_data.dmnd`, and the DiamondScore uses
    the release's training `prop_annotations`.
- **Ontology and theories**: GO 2021-11-16; vocabulary theories `dgp2_go_cc` and `dgp2_go_mf`
  exactly as built in P4.1a (alias / replaced_by mapping; duplicate heads merged by max).
- **Fitting data**: every DeepGOPlus 1.0.6 test protein with at least one non-root label in the
  namespace: 2,912 for GO cc, 2,129 for GO mf. All methods are refit on all of these proteins
  (no 50/50 split).
- **Reproducibility gates**, which must pass before any new-target scoring:
  - (a) Recomputed CNN scores for the 3,874 old test proteins match `predictions.pkl` to a max
    abs difference ≤ 1e-4.
  - (b) DIAMOND hits recomputed for the old test sequences are compared with `test_diamond.res`.
    The agreement is reported. Fused scores are used in the confirmation only if the recomputed
    old fused scores match the release-based ones to a median abs difference ≤ 1e-3 and the
    old-set metrics change by < 0.005 Fmax. Otherwise only the CNN input is confirmed.

## 3. New targets (new side; used only for evaluation)

- **New snapshot**: Swiss-Prot 2024_01 with GO 2024-01-17.
- **Target set for namespace ns ∈ {cc, mf}**: entries meeting all of the following.
  1. ≥ 1 experimental annotation in ns in 2024_01. Experimental means one of the DeepGOPlus
     EXP_CODES: EXP, IDA, IPI, IMP, IGI, IEP, TAS, IC, HTP, HDA, HMP, HGI, HEP.
  2. **No experimental GO annotation in any namespace in Swiss-Prot 2021_04** under any of the
     entry's accessions (primary or secondary). This is the CAFA "no-knowledge" setting and
     comes from comparing the two snapshots, not from entry update dates.
  3. None of the entry's accessions or protein names occurs in the DeepGOPlus 1.0.6 training or
     test data.
  4. Its sequence is not identical to any sequence in the DeepGOPlus training data or in the
     fitting data. Overlap counts are reported.
  5. After label construction (§4), it has ≥ 1 non-root positive label in the ns vocabulary.
- **Missing annotations**: unannotated terms count as negatives, following the CAFA convention.
  This is a known limitation, because absence of annotation is not a reliable negative. It
  affects all methods equally and is stated in the paper.

## 4. Label construction for new targets (mirrors the old labels)

The 2024_01 experimental annotations in ns are propagated in GO 2024-01-17 over is_a plus all
relationship types (DeepGOPlus `Ontology(with_rels=True)`). The results are mapped to GO
2021-11-16 ids (alt_id / unique replaced_by, using both releases' maps) and restricted to the
ns vocabulary. They are then closed upward in the `dgp2_<ns>` theory, as in P4.1a.

## 5. Methods (all hyperparameters fixed now)

- **No fitting**: raw, max_closure, isotonic (Hildreth, tol 1e-9).
- **Fitted on the old data, then isotonic projection**:
  - depth_platt_iso (2 params / depth);
  - term_platt_iso (2 params / term, ridge 1e-2, min 3 positives);
  - neighbor_lr_iso ([1, l, max/min parent l, max child l], 5 / depth);
  - same_features_lr_iso ([1, l, m], 3 / depth);
  - theta_star_feature_lr_iso ([1, l, m, θ\*/10], 4 / depth);
  - whole_vector_pca32_lr_iso (32 principal components fitted on old logits + [1, l, m]).
  All ridge 1e-3, L-BFGS.
- **C2**: target μ\* = 0.999·isotonic + 0.001·μ_unif, where μ_unif = exact marginals at p = 0.5
  (as in P4.5). θ\* by exact moment matching (L-BFGS, gtol 1e-11, maxiter 5000). Residual
  features [1, l, m] per depth, ridge 1e-3, L-BFGS maxiter 1000, initial w = 0. Exact marginals
  from the d4 circuit using the stable log-partition.

## 6. Evaluation

- The namespace root is excluded.
- **Primary metrics**: protein-centric Fmax (thresholds 0.01..0.99) and micro average precision
  (tie-corrected).
- **Secondary metrics**: Brier, per-label log-loss, ECE (15 bins).
- **Inference**: 1,000 paired bootstrap resamples of target proteins (seed 0). Report the
  difference C2 − method for every method and every metric.

## 7. Pre-registered hypotheses and decision

- **H1**: C2 > isotonic on both primary metrics (95% CI > 0), for each confirmed
  (namespace, input) setting.
- **H2**: C2 > every non-WMC fitted baseline of §5 on both primary metrics.
- The C2 line is called **confirmed** if H1 holds in ≥ 3 of the confirmed settings and H2 holds
  in ≥ 2. Otherwise it is **not confirmed**, and the exploratory gains are reported as such.
  All results are reported whatever the outcome.
