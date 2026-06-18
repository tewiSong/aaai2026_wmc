"""Brute-force exact WMC by full enumeration, for validating the SDD and modular paths.

Only usable for small variable counts (<= ~22), this enumerates all 2^n assignments,
keeps the models of the theory, and computes exact marginals as weighted ratios. It is
the ground-truth reference invoked in the validation checks; it uses no heuristics.
"""

from __future__ import annotations

from itertools import product
from typing import List, Sequence, Tuple


def satisfies(assignment: Sequence[int], clauses: Sequence[Tuple[int, ...]]) -> bool:
    """assignment[i] in {0,1} is the truth of variable (i+1)."""
    for clause in clauses:
        ok = False
        for lit in clause:
            v = abs(lit) - 1
            want = 1 if lit > 0 else 0
            if assignment[v] == want:
                ok = True
                break
        if not ok:
            return False
    return True


def brute_force_marginals(
    n_vars: int,
    clauses: Sequence[Tuple[int, ...]],
    priors: Sequence[float],
) -> Tuple[float, List[float]]:
    """Return (Z, marginals) by exhaustive enumeration."""
    if n_vars > 24:
        raise ValueError(f"brute force refused for n_vars={n_vars} (> 24)")
    z = 0.0
    num = [0.0] * n_vars
    for assign in product((0, 1), repeat=n_vars):
        if not satisfies(assign, clauses):
            continue
        w = 1.0
        for i in range(n_vars):
            w *= priors[i] if assign[i] == 1 else (1.0 - priors[i])
        z += w
        for i in range(n_vars):
            if assign[i] == 1:
                num[i] += w
    return z, [n / z for n in num]
