# P7 protocol: locked GO bp time-split confirmation of core C2 with a per-term residual

Written by Claude on 2026-10-25. It is frozen before any evaluation on the P7 targets. Its sha256
is recorded in `results/p7_protocol_lock.txt`, and `scripts/p7_3_go_bp_locked_time_split_evaluation.py`
checks that hash before it runs. Any later deviation is recorded as a separate, dated amendment.

## 0. Why a new confirmation, and what was seen before locking

- P5 confirmed C2 on GO cc / GO mf (H1 3/4, H2 2/4). Afterwards, P6.4 (exploratory split) found
  that C2's deficit to term Platt on CNN inputs comes from its depth-shared residual. Adding a
  per-term affine residual fixed it: AP was significantly better than term Platt in 4/4 settings.
- That model was designed after the P5 results, so the P5 targets cannot confirm it. GO bp time-split
  targets have never been evaluated, so they are the clean confirmation set.
- Seen before locking: the exploratory GO bp results of P6.3 (old test proteins only), the P6.5
  truncation-sensitivity check, and the P7.1 / P7.2 target counts and reproducibility gates.
  No P7 target score has been compared with a P7 label.
- **P6.5 was still running at lock time.** It compares the P6.3 cores with larger cores
  (τ ≤ 0.007) at fixed weights. The core rule of §4 is fixed by feasibility: the larger cores
  cannot be computed for all 5,650 proteins. So P6.5 decides only how the GO bp results are
  described — insensitive to truncation, or the exact posterior on the core with a truncation
  approximation outside it. It does not change this protocol.
- **Smoke test before locking**: 60 old + 60 new fused proteins, with no lock. It checked the code
  path only; its metrics are not informative at that size. On its 48-protein inner split λ
  reached the top of the then-planned grid {1e-3, …, 1}, so 10 was added to the grid before locking.
- **Choice of the primary metric.** P5 required both Fmax and micro AP. On GO bp, every method's
  Fmax lies within 0.531–0.541 (fused) and 0.412–0.417 (CNN) in P6.3, so Fmax is not expected to
  separate the methods. The claim tested here is therefore a ranking-quality claim:
  - micro AP is the primary metric;
  - Fmax enters as a non-inferiority condition (H3).

  This choice is made before seeing any P7 result, and it is stated in the paper.

## 1. Question

On proteins whose GO bp annotations appeared only after the predictor and the fitting data were
frozen, does core C2 with a per-term affine residual improve micro AP over isotonic regression and
over every fitted non-WMC baseline, without losing Fmax?

## 2. Frozen information (old side)

- **Predictor, CNN code, DIAMOND, fused formula and reproducibility gates**: as in the P5 protocol
  §2 and amendment 1 (truncation to MAXLEN), with α_bp from `metadata/last_release.json`.
- **Gates**: gate (a) is re-run in P7.2a. Gate (b) is evaluated for GO bp in P7.2b with the P5
  thresholds (median abs diff ≤ 1e-3, old-set Fmax change < 0.005). If gate (b) fails, only CNN is
  confirmed.
- **Ontology and theory**: GO 2021-11-16; vocabulary theory `dgp2_go_bp` exactly as built in
  P4.1a (4,475 terms).
- **Fitting data**: every DeepGOPlus 1.0.6 test protein with ≥ 1 non-root bp label (2,973). All
  methods are fitted on all of them.

## 3. New targets and labels

- Built by P7.1 with the P5 rules (§3–§4 of the P5 protocol), for namespace bp:
  - ≥ 1 experimental bp annotation in Swiss-Prot 2024_01;
  - no experimental GO annotation in any namespace in 2021_04;
  - no accession / name overlap and no identical sequence with DeepGOPlus 1.0.6 data;
  - ≥ 1 non-root label in the vocabulary.
- Labels are propagated in GO 2024-01-17 (is_a + all relationships), mapped to GO 2021-11-16, and
  closed upward in `dgp2_go_bp`.
- Unannotated terms count as negatives (CAFA convention; known limitation).

## 4. Inference for the C2 models (fixed)

- **Target**: μ\* = 0.999·isotonic + 0.001·μ_ref, with the descendant-fraction reference
  μ_ref_i = (|desc(i)| + 1)/(n + 1).
- **Adaptive hot cores** (P6.3), per protein, old and new alike:
  - H(τ) = {μ\*_i ≥ τ} ∪ {root};
  - τ is the smallest value on the ladder {0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99}
    such that |H| ≤ 1,500, d4 compiles within 120 s, and the circuit has ≤ 2,000,000 edges;
  - if no level fits, the core is the root only.
- The model is the exact posterior on H given the cold terms off. Cold terms take μ\*, then the
  order is enforced exactly (child ≤ parent).
- θ\*_H is obtained by exact moment matching on the core circuit (L-BFGS, gtol 1e-11, maxiter 5000).

## 5. Methods (all hyperparameters fixed now)

- **No fitting**: raw, max_closure, isotonic.
- **Fitted then isotonic**: depth_platt_iso, term_platt_iso, neighbor_lr_iso,
  same_features_lr_iso, theta_star_feature_lr_iso (θ\* = core θ\*, logit μ\* on cold terms),
  whole_vector_pca32_lr_iso. All are identical to P5 / P6.3.
- **C2_term_affine (primary)**:
  - parameters: θ_i = θ\*_i + w_{d,0} + w_{d,1}·l_i + w_{d,2}·m_i + b_i + a_i·l_i on the core,
    with l = logit(score) and m = logit(μ\*);
  - ridge 1e-3 on w; ridge λ on (a, b), with λ ∈ {1e-3, 1e-2, 1e-1, 1, 10};
  - λ is chosen by the mean joint conditional NLL on an inner 80/20 split of the old proteins
    (seed 1). Each inner fit starts at the C2 weights fitted on all old proteins, with a = b = 0;
  - the model is then refitted on all old proteins with the chosen λ;
  - L-BFGS, maxiter 1000.
- **C2 (secondary)**: the depth-shared residual only (w), ridge 1e-3, L-BFGS maxiter 1000.

## 6. Evaluation

- The root is excluded.
- **Primary metric**: micro average precision.
- **Non-inferiority metric**: protein-centric Fmax.
- **Secondary metrics**: Brier, log-loss, ECE (15 bins).
- **Inference**: 1,000 paired bootstrap resamples of targets (seed 0). The difference
  model − method is reported for both C2 models, against every method and on every metric.

## 7. Pre-registered hypotheses and decision (for C2_term_affine, per confirmed input)

- **H1**: C2_term_affine − isotonic, micro AP, 95% CI lower bound > 0.
- **H2**: C2_term_affine − each of the 6 fitted baselines, micro AP, 95% CI lower bound > 0.
- **H3** (non-inferiority): C2_term_affine − isotonic and − each fitted baseline, Fmax, 95% CI
  lower bound > −0.005.
- **Decision**:
  - **confirmed**: H1, H2 and H3 hold in every confirmed input (CNN, plus fused if gate (b) passes);
  - **partially confirmed**: H1 and H3 hold in every confirmed input, and H2 holds in at least one;
  - **not confirmed**: otherwise.

  The same quantities are reported for C2 (secondary) without a decision. All results are
  reported whatever the outcome.
