"""truepath: modular weighted model counting for the GO true-path theory.

This package implements the inference layer studied in the paper:
  - the true-path theory over a Gene Ontology namespace (is-a / part-of backbone
    plus EL++ normal-form conjunction and existential definitions),
  - the soft closure (soft-OR loopy belief propagation),
  - exact weighted model counting (WMC) marginals via knowledge compilation (SDD),
  - min-fill treewidth and Shafer-Shenoy junction-tree calibration,
  - the modular exact-WMC construction (per-module SDD + weighted separator messages),
  - the rank-r separator (tensor-train) dial for the residual reconvergence core.

No fallback / approximate shortcuts are used in place of the exact computations:
every "exact" routine is exact and is validated against brute-force enumeration on
small instances.
"""
