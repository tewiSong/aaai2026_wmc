# Modular WMC for NeSy over GO — experiment code

Reproduces the experiments of *Modular weighted model counting for neuro-symbolic
inference over large ontologies* on the Gene Ontology (GO).

## Environment

```
conda activate /ibex/user/songt/conda_envs/nesy
```
Key packages already present: `torch 2.3.1+cu121`, `pysdd`, `problog`, `networkx`,
`numpy/scipy`, `tensorly`, `matplotlib`, Java 11 (for reference). No fallback or simplified
implementations are used: every exact routine is exact and is validated against brute force.

## Data

Symlinked under `data/` (raw inputs live on `/ibex/user/songt/datasets/`):
- `data/go-basic.obo` -> GO OBO release (is-a / part-of hierarchy, namespace tags).
- `data/go.owl`       -> GO OWL (logical definitions; inspected for axiom content).
- `data/processed/`   -> `/ibex/user/songt/datasets/nesy2026/go_truepath/` (cached theories).

The true-path theory per namespace is the EL++ subsumption backbone (is-a) built from
`go-basic.obo`; part-of is an existential role (off by default, as in the EL++ normal form).
Three namespaces: cellular_component (cc), molecular_function (mf), biological_process (bp).

## Library (`src/truepath/`)

- `graph.py`          OBO parser, `TruePathTheory`, constraint-graph construction.
- `treewidth.py`      min-fill treewidth (lazy-heap, feasible on 26k-node bp).
- `softclosure.py`    soft closure = loopy sum-product belief propagation (exact on trees).
- `exact_wmc.py`      exact WMC marginals via SDD compilation (PySDD).
- `junction_tree.py`  exact WMC marginals via Shafer-Shenoy / Hugin junction-tree calibration.
- `modular.py`        core-cut trade-off curve + modular exact WMC over modules.
- `rank_dial.py`      tensor-train rank-r boundary dial for the residual core.
- `bruteforce.py`     full-enumeration validator (ground truth for small instances).

All exact paths agree to machine precision (validated on the diamond, chains, and random DAGs).

## Experiments (`scripts/`)

- `00_prepare_data.py`              dataset statistics (terms, edges, multi-parent fraction).
- `01_exp1_bp_reconvergence.py`     prior battery (tree-exactness + diamond inexactness) and
                                    cc exact-vs-soft-closure error, localized at reconvergences,
                                    with the error-vs-in-degree regression and figure.
- `02_exp2_modular_wmc.py`          `--part {treewidth,corecut,modular,rankdial,all}`:
                                    treewidth per namespace; core-cut trade-off curve (mf/bp);
                                    modular exact WMC runtimes (validated); rank-r dial table +
                                    40-core statistics.
- `03a_exp3_compile_wall.py`        exact knowledge-compilation wall: exact-WMC cost is 2^tw
                                    (junction-tree clique table); compiles cc, not mf/bp.
- `03b_exp3_scallop_topk.py`        top-k provenance (Scallop) drift vs k at reconvergences.
- `03c_exp3_carrychain.py`          carry-chain (N-digit MNIST addition), two modes (GPU):
                                    `exact` IA marginal (convolution) degrades gently with N;
                                    `sample` Monte-Carlo WMC estimate (the A-NeSI/NeSyDM
                                    starting point) collapses as the carry couples more digits.

## Running on SLURM (Ibex)

```
cd slurm
sbatch run_cpu_main.sbatch        # Exp0, Exp1, Exp2(treewidth+rankdial), Exp3a, Exp3b  (~hours, partition=batch)
sbatch run_cpu_modular.sbatch     # Exp2 core-cut curve + modular exact WMC for mf/bp     (long, partition=batch)
sbatch run_gpu_carrychain.sbatch  # Exp3c carry-chain                                     (partition=gpu, 1 GPU)
```
Results are written to `results/` as JSON/CSV plus the cc regression figure.

## Reproduction status (login-node smoke runs)

- Battery: single-edge / star / polytree exact across the 9x9 prior grid; diamond inexact.
- cc: exact JT in ~8s; soft closure mean error 2.0e-4, max 6.4e-2; multi-parent error 6.0x
  single-parent (paper: 7x).
- max in-degree per namespace 4 / 7 / 9 (paper: 4 / 7 / 8).
- Scallop four-node: 0.50 / 0.875 / 0.9375 at k = 1 / 3 / 10 (matches paper).
- Exact WMC cost is 2^tw: monolithic exact compiles cc (tw 12, ~3s) but walls on mf (2^37) and
  bp (2^269); succeeds to tw~23, exhausts memory by tw~62. Modular caps modules at tw<=13.
- rank-r dial: rank-1 KL ~1.5 nats median over 40 cc cores, dials to exact below dense 2^w.

Note: term counts and treewidth differ slightly from the paper (different GO release:
go-basic 2025-06 here); the structural and qualitative results reproduce.

## Lean 4 formalization (`lean/`)

Two developments verify Theorem 1 and the reconvergence claim:
- `lean/Wmc.lean` — Lean-core (no mathlib), custom rational `Q = Int×Int`, every theorem
  closed by `decide` (kernel-checked), axiom footprint `{propext}` only. Proves
  `single_edge_exact`, `star2_exact`, `star3_exact` (soft-OR = exact WMC on the tree fragment),
  `diamond_exact_5_6`, `diamond_softor_9_10`, `diamond_softor_ne_exact` (5/6 vs 9/10).
  Build:  `ELAN_HOME=/ibex/user/songt/elan /ibex/user/songt/elan/bin/lean lean/Wmc.lean`
- `lean/wmcmath/` — parametric mathlib proof `WmcStar.star_marginal_eq_softOR`, over all
  `k : ℕ` and all real priors; footprint `{propext, Classical.choice, Quot.sound}`, no `sorry`.

Toolchain: Lean 4.31 via elan at `/ibex/user/songt/elan` (`ELAN_HOME=/ibex/user/songt/elan`).
IMPORTANT: the mathlib build/cache must live on the weka `/ibex` filesystem, not `/home` NFS
(leantar decompression of mathlib's thousands of small oleans corrupts on NFS). Build the
mathlib proof from `/ibex/user/songt/wmcmath` with:
```
export ELAN_HOME=/ibex/user/songt/elan PATH=/ibex/user/songt/elan/bin:$PATH
export XDG_CACHE_HOME=/ibex/user/songt/.cache
lake exe cache get      # fetches prebuilt mathlib oleans (on weka)
lake build Wmcmath
```
Use the targeted imports in `Wmcmath/Basic.lean` (not `import Mathlib`, which is slow to load).
