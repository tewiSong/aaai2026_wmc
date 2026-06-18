"""Exact weighted model counting (WMC) marginals via knowledge compilation (SDD).

We compile a propositional theory given as Horn clauses to a Sentential Decision Diagram
with PySDD, then read exact per-variable marginals

    mu_i = WMC(T & y_i; p) / WMC(T; p)

directly from the weighted-model-counting pass (`WmcManager.literal_pr`). This is the
exact object the soft closure approximates. The vtree is built from a supplied variable
order (the min-fill elimination order) so the compiled circuit size matches the measured
treewidth.

No approximation is used anywhere here: the SDD is an exact representation of T and the
WMC pass is exact arithmetic. Results are validated against brute force in `bruteforce.py`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from pysdd.sdd import SddManager, Vtree


def compile_clauses(
    n_vars: int,
    clauses: Sequence[Tuple[int, ...]],
    var_order: Optional[Sequence[int]] = None,
    vtree_type: str = "balanced",
):
    """Compile a CNF (list of signed-index clauses) into an SDD.

    Variables are 1..n_vars; a clause is a tuple of nonzero signed indices.
    Returns (manager, root_node).
    """
    if var_order is None:
        var_order = list(range(1, n_vars + 1))
    else:
        var_order = list(var_order)
    assert sorted(var_order) == list(range(1, n_vars + 1)), "var_order must be a permutation of 1..n"

    vtree = Vtree(var_count=n_vars, var_order=var_order, vtree_type=vtree_type)
    mgr = SddManager.from_vtree(vtree)
    mgr.auto_gc_and_minimize_off()

    root = mgr.true()
    for clause in clauses:
        lits = [mgr.literal(l) for l in clause]
        disj = lits[0]
        for lit in lits[1:]:
            disj = disj | lit
        root = root & disj
    return mgr, root


def wmc_marginals(
    mgr: SddManager,
    root,
    priors: Sequence[float],
) -> Tuple[float, List[float]]:
    """Return (Z, marginals) where Z = WMC(T;p) and marginals[i] = P(y_{i+1}=1 | T).

    priors[i] is the prior probability that variable (i+1) is true.
    """
    n_vars = len(priors)
    wmc = root.wmc(log_mode=False)
    for i in range(n_vars):
        v = i + 1
        wmc.set_literal_weight(mgr.literal(v), priors[i])
        wmc.set_literal_weight(mgr.literal(-v), 1.0 - priors[i])
    z = wmc.propagate()
    marg = [wmc.literal_pr(mgr.literal(i + 1)) for i in range(n_vars)]
    return float(z), [float(m) for m in marg]


def exact_marginals_for_clause_set(
    n_vars: int,
    clauses: Sequence[Tuple[int, ...]],
    priors: Sequence[float],
    var_order: Optional[Sequence[int]] = None,
) -> Tuple[float, List[float], int]:
    """Compile + count. Returns (Z, marginals, sdd_size)."""
    mgr, root = compile_clauses(n_vars, clauses, var_order=var_order)
    z, marg = wmc_marginals(mgr, root, priors)
    size = root.size() if hasattr(root, "size") else mgr.size()
    return z, marg, int(size)
