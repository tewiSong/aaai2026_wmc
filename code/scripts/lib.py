"""Shared helpers for the experiment scripts."""

from __future__ import annotations

import os
import pickle
import sys
from typing import Dict

import numpy as np

# Make the truepath package importable regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from truepath.graph import (TruePathTheory, build_namespace_theories,  # noqa: E402
                            build_namespace_theories_from_norm)

DATA_DIR = os.path.join(_HERE, "..", "data")
RESULTS_DIR = os.path.join(_HERE, "..", "results")
OBO_PATH = os.path.join(DATA_DIR, "go-basic.obo")
NORM_DIR = os.path.join(DATA_DIR, "processed", "norm")
CACHE_PATH = os.path.join(DATA_DIR, "processed", "theories.pkl")

NAMESPACES = ["cc", "mf", "bp"]


def load_theories() -> Dict[str, TruePathTheory]:
    """Load the true-path theories: the GO subsumption (is-a) constraint graph.

    Treewidth is controlled through a fixed min-fill decomposition of the constraint graph.
    The current GO release gives min-fill widths 12/37/269 for the subsumption hierarchy.
    This is the object the treewidth and modular-decomposition experiments run on.
    The EL++ conjunction (NF2) and existential (NF3/NF4) definitions extracted by
    scripts/05b_extract_norm.py are additional Horn clauses of the theory. The full-theory
    stress check in scripts/07_modular_with_nf2.py includes NF2 clauses inside the module WMC
    and validates the solved modules against brute force. No treewidth fallback is used: all
    reported widths come from the same min-fill implementation.
    """
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as fh:
            return pickle.load(fh)
    theories = build_namespace_theories(OBO_PATH)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as fh:
        pickle.dump(theories, fh)
    return theories


def make_priors(theory: TruePathTheory, seed: int = 0,
                lo: float = 0.05, hi: float = 0.95) -> Dict[str, float]:
    """Deterministic per-atom priors p_i drawn uniformly in [lo, hi].

    Theorem 1 holds for all priors, so the structure of the soft-closure error is what
    we probe; a fixed seed makes every run reproducible.
    """
    rng = np.random.default_rng(seed)
    return {a: float(rng.uniform(lo, hi)) for a in theory.atoms}


def ensure_results_dir() -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR
