"""Rank-r separator (tensor-train) dial for the residual reconvergence core.

At a residual core the soft closure represents the boundary joint as a rank-1 (product /
independent) message, which is the independence assumption at the boundary. We measure how
far that is from the exact constraint-induced boundary joint q*, and how a rank-r message
(a tensor train over the boundary atoms) dials the residual error to zero.

For a small core sub-theory we:
  - build q* exactly by enumerating the satisfying assignments weighted by the priors,
  - form the independent (rank-1) product of its marginals (the soft-closure / IA head),
  - compute KL(q* || .) and total variation for rank-1 and for rank-r tensor-train
    truncations, and the exact tensor-train rank at which the joint is reproduced.

TT-SVD is implemented directly (a sequence of SVD truncations); no approximation library
is required and the exact-rank path uses a numerical-zero tolerance only to count
vanishing singular values. Low-rank SVD truncation is not guaranteed to be nonnegative;
when a probability message is required we explicitly project the reconstructed tensor
onto the probability simplex by removing negative mass and renormalizing, and report the
amount of removed negative mass in the result rows.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .graph import TruePathTheory


def boundary_joint(theory: TruePathTheory, priors: Dict[object, float]) -> Tuple[np.ndarray, List[object]]:
    """Exact constraint-induced joint over the theory's atoms as a tensor of shape (2,)*w.

    The atoms are the core/boundary atoms; clauses are the constraints among them. The
    joint is q*(y) proportional to [y satisfies T] * prod_i p_i^{y_i}(1-p_i)^{1-y_i}.
    """
    atoms = theory.atoms
    w = len(atoms)
    if w > 22:
        raise ValueError(f"boundary_joint refused for w={w} (> 22)")
    clauses = theory.clauses()
    tensor = np.zeros((2,) * w, dtype=np.float64)
    pri = [priors[a] for a in atoms]
    for assign in product((0, 1), repeat=w):
        ok = True
        for clause in clauses:
            sat = False
            for lit in clause:
                v = abs(lit) - 1
                want = 1 if lit > 0 else 0
                if assign[v] == want:
                    sat = True
                    break
            if not sat:
                ok = False
                break
        if not ok:
            continue
        wt = 1.0
        for i in range(w):
            wt *= pri[i] if assign[i] == 1 else (1.0 - pri[i])
        tensor[assign] = wt
    s = tensor.sum()
    if s <= 0:
        raise RuntimeError("boundary joint has zero mass; constraints unsatisfiable")
    return tensor / s, atoms


def product_marginal(tensor: np.ndarray) -> np.ndarray:
    """Rank-1 independent approximation: outer product of the per-axis marginals."""
    w = tensor.ndim
    marg = []
    for ax in range(w):
        other = tuple(k for k in range(w) if k != ax)
        m = tensor.sum(axis=other)
        marg.append(m)
    out = np.ones((1,))
    p = marg[0]
    for ax in range(1, w):
        p = np.multiply.outer(p, marg[ax])
    return p


def tt_svd(tensor: np.ndarray, max_rank: int = None, tol: float = 1e-12) -> List[np.ndarray]:
    """Tensor-train decomposition by sequential SVD.

    Returns the list of TT cores. If `max_rank` is given, every bond is truncated to at
    most that rank; otherwise bonds keep all singular values above `tol` (exact TT).
    """
    shape = tensor.shape
    w = tensor.ndim
    cores: List[np.ndarray] = []
    c = tensor.reshape(1, -1)
    r_prev = 1
    for k in range(w - 1):
        nk = shape[k]
        c = c.reshape(r_prev * nk, -1)
        U, S, Vt = np.linalg.svd(c, full_matrices=False)
        if max_rank is not None:
            r = min(max_rank, np.sum(S > tol))
        else:
            r = int(np.sum(S > tol))
        r = max(r, 1)
        U = U[:, :r]
        S = S[:r]
        Vt = Vt[:r, :]
        cores.append(U.reshape(r_prev, nk, r))
        c = (np.diag(S) @ Vt)
        r_prev = r
    cores.append(c.reshape(r_prev, shape[-1], 1))
    return cores


def tt_reconstruct(cores: List[np.ndarray]) -> np.ndarray:
    """Contract TT cores back into a full tensor."""
    out = cores[0]  # (1, n0, r0)
    for k in range(1, len(cores)):
        out = np.tensordot(out, cores[k], axes=([out.ndim - 1], [0]))
    # out shape: (1, n0, n1, ..., 1) -> squeeze the boundary singleton dims
    out = out.reshape(out.shape[1:-1])
    return out


def tt_max_rank(cores: List[np.ndarray]) -> int:
    return max(core.shape[2] for core in cores[:-1]) if len(cores) > 1 else 1


def probability_tensor(
    tensor: np.ndarray,
    *,
    project_negative: bool,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return a normalized probability tensor and diagnostics.

    Exact-rank reconstructions should only have roundoff-level negative entries. Truncated
    TT-SVD can produce signed tensors; when `project_negative` is true, this function
    applies the explicit nonnegative projection used by the rank-dial experiment and
    records how much negative mass was removed. When it is false, significant negative
    mass is an error rather than a hidden repair.
    """
    arr = np.asarray(tensor, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError("rank message contains non-finite entries")
    min_value = float(arr.min())
    negative = arr < -tol
    negative_mass = float(-arr[negative].sum()) if np.any(negative) else 0.0
    pre_sum = float(arr.sum())
    projected = False
    if np.any(negative):
        if not project_negative:
            raise FloatingPointError(
                f"rank message has negative probability mass {negative_mass:.3e}"
            )
        arr = np.maximum(arr, 0.0)
        projected = True
    else:
        arr = np.where(arr < 0.0, 0.0, arr)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("rank message has non-positive mass after normalization")
    return arr / total, dict(
        projected=projected,
        pre_projection_sum=pre_sum,
        projected_negative_mass=negative_mass,
        min_value_before_projection=min_value,
    )


def kl_tv(q: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """KL(q || p) in nats and total variation, over the support of q."""
    qf = q.ravel()
    pf = p.ravel()
    kl = 0.0
    tv = 0.0
    for i in range(qf.size):
        tv += abs(qf[i] - pf[i])
        if qf[i] > 0:
            if pf[i] <= 0:
                kl = np.inf
            else:
                kl += qf[i] * np.log(qf[i] / pf[i])
    return float(kl), float(0.5 * tv)


def rank_dial(theory: TruePathTheory, priors: Dict[object, float],
              ranks: Sequence[int]) -> Dict:
    """Compute the rank-r dial table for one core.

    Returns dict with: w (atoms), n_models, exact_rank, and per-rank (kl, tv, gap_closed),
    where rank-1 is the independent product and gap_closed is the fraction of the rank-1
    total variation removed.
    """
    q, atoms = boundary_joint(theory, priors)
    n_models = int(np.sum(q > 0))
    # exact TT rank
    exact_cores = tt_svd(q, max_rank=None, tol=1e-12)
    exact_rank = tt_max_rank(exact_cores)

    # rank-1 = independent product
    p1 = product_marginal(q)
    kl1, tv1 = kl_tv(q, p1)

    rows = []
    for r in ranks:
        if r == 1:
            kl, tv = kl1, tv1
            diag = dict(projected=False, pre_projection_sum=float(p1.sum()),
                        projected_negative_mass=0.0,
                        min_value_before_projection=float(p1.min()))
        else:
            cores = tt_svd(q, max_rank=r, tol=1e-12)
            pr, diag = probability_tensor(tt_reconstruct(cores),
                                          project_negative=(r < exact_rank),
                                          tol=1e-12)
            kl, tv = kl_tv(q, pr)
        gap = (tv1 - tv) / tv1 if tv1 > 0 else 0.0
        rows.append(dict(rank=r, kl=kl, tv=tv, gap_closed=gap, **diag))
    return dict(w=len(atoms), n_models=n_models, exact_rank=exact_rank,
                kl_rank1=kl1, tv_rank1=tv1, rows=rows)
