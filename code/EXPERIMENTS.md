# Experiments design — Modular WMC for NeSy over large ontologies

Venue: **NeSy 2026** (PMLR/JMLR, double-blind, 10-page body limit). This paper is a
**theory / inference-layer** paper, not an application benchmark. Protein function
prediction (CAFA) is deliberately deferred to the **companion paper** and is NOT a main
experiment here.

## Guiding principles

- **Gold standard**: exact WMC marginal `mu_i = WMC(T & y_i) / WMC(T)`. All errors are
  reported relative to it.
- **Unified baseline ladder** (used throughout):
  1. Raw independent scores (no correction; lower bound)
  2. Hard true-path closure (upward max propagation; logical closure)
  3. Soft closure / loopy BP (what is actually deployed)
  4. Monolithic exact WMC (SDD / junction tree; only on cc and small modules, as gold ref)
  5. **Ours**: Modular exact WMC
  6. **Ours**: Modular WMC + rank-r separator
- **Core metrics**: marginal error vs exact WMC (mean / p95 / max); error localization at
  reconvergences; fraction of terms exactly computable; core size; runtime / memory;
  rank–error tradeoff.
- **Page budget**: body holds main tables + key figures. Prior battery, full rank-dial
  table, and the SROIQ synthetic fragment go to the appendix.

---

## Exp 0 — Setup and reproducibility (half paragraph + table)

- Data: GO `go.obo` / `go.norm`; cc 2.8k / mf 6.9k / bp 21k terms; graph built from
  EL++ normal forms.
- Gold check: on small modules, modular-exact vs brute force agree to 1e-16.
- State hardware, repeat counts, min-fill settings in one line.

## Exp 1 — Soft closure = BP; error confined to reconvergences (Contribution 1)

**Claim**: on the tree fragment the soft closure equals WMC for all priors; error only at
reconvergences.

- **1a Lean formalisation**: axiom footprint (`propext / Classical.choice / Quot.sound`
  only; no `sorry` / `native_decide`); 9x9 prior battery (single-edge and multi-parent
  polytree exact, diamond wrong everywhere). -> qualitative correctness.
- **1b cc empirical** (key): cc treewidth 9, so exact junction tree is feasible.
  - Main table: baseline ladder 1-5 with mean / p95 / max error vs exact, fraction of
    terms off by >0.01, runtime.
  - **Key figure**: error at multi-parent (reconvergent) terms vs single-parent terms
    (~7x). Improvement: plot **error vs node in-degree regression** to turn the "7x" from
    two points into a continuous relation (strengthens the theorem's prediction).

## Exp 2 — Treewidth + modular exact WMC (Contribution 2)

**Claim**: bp is infeasible monolithically; cutting a <8% core leaves 92-100% exactly
computable; the residual core is rank-r dialable to exact.

- **2a Treewidth measurement**: cc/mf/bp = 9/31/272; argue bp admits no monolithic exact.
- **2b Core-cut trade-off curve** (improvement): greedily delete highest-degree nodes;
  plot the full "fraction cut -> fraction exactly computable -> time" curve, not just the
  two points (mf 2.95% / bp 7.7%).
- **2c Modular construction, run not inferred**: module count, max module treewidth,
  wall-clock for all marginals (mf 488 modules 1.1s; bp 2210 modules 12.8s), verified to
  machine precision.
- **2d rank-r dial**: real cc core (13 terms) KL / TV / gap-closed vs rank (small table in
  body); 40-core statistics (median gap 1.41 nats, median rank 11) in appendix. Add a
  column for **rank vs end-to-end WMC error** (currently only KL/TV; downstream impact is
  missing).

## Exp 3 — NeSy framework scalability comparison (Contribution 3)

**Claim**: each existing framework trades correctness for scale in one way; ours is alone
in being both scalable and accuracy-characterized.

- **Controlled input**: DeepProbLog / Scallop encode the *identical* GO true-path
  inference; first validate they reproduce the exact marginals, then test scalability.
- Data: small GO induced subgraphs + cc/mf/bp.
- Main table baselines: DeepProbLog (exact compilation, walls ~400 terms), Scallop (top-k,
  full bp in 0.9s but drifts at reconvergences; report k=1/3/10 recovery curve),
  SPL / exact circuit (small graphs / cc), soft closure, **ours**.
- **A-NeSI / NeSyDM NOT in the GO main table**; separate Exp 3b synthetic stress test:
  carry-chain, three seeds, N=1->4 collapse (0.97->0.23, 0.96->0.03-0.13).
- Summary table: `scales to full GO?` / `accuracy at scale`.

## Appendix — SROIQ outlook experiment

Controlled disjunctive fragment (w=6 atoms, 5-mode multimodal joint): IA discards 2.08
nats, exact only at rank 3 (table). **Explicitly labelled** synthetic, not a GO result;
serves as evidence for the future-work direction, not a main experiment.

---

## Concrete change list vs current manuscript

1. Add explicit **raw / hard closure** baseline rows in the cc empirical (Sec. 4) and
   throughout Sec. 5.
2. Sec. 5 core cut: expand the two points into a full **trade-off curve**.
3. Sec. 4: add the **error vs in-degree regression** figure.
4. Sec. 5 rank dial: add a **rank vs end-to-end WMC error** column.
5. Do **not** add the protein/CAFA experiment (companion paper).
6. Keep error-vs-exact as the primary metric; do **not** introduce Fmax / ECE /
   violation-rate as primary metrics for this paper.

## Decision rationale (why this scoping)

- A second AI proposed three main tasks including protein function prediction (CAFA) and
  framed the paper for AAAI with Fmax/AUPR/ECE metrics. That framing is rejected here:
  the manuscript is NeSy 2026 with a 10-page limit and explicitly defers protein function
  prediction to a companion paper (abstract + Sec. 6.1). Adding it would blow the page
  budget, undercut the companion paper, and change the paper's identity.
- What was adopted from that proposal: the **baseline ladder** (raw / hard closure as
  ablation floors), which the original outline lacked.
