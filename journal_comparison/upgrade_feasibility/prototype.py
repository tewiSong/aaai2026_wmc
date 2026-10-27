"""Small, reproducible feasibility experiments, NOT the author's implementation.

Dependencies: Python >=3.10 and numpy. Run with the bundled Python runtime.
Dense enumeration below is an independent verification oracle on small instances;
it is not claimed to scale to biomedical ontologies.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
from decimal import Decimal, localcontext
import numpy as np

ROOT = Path(__file__).resolve().parent


def bits(n):
    return ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(int)


def product_weights(x, p):
    return np.prod(np.where(x, p, 1 - p), axis=1)


@dataclass
class Module:
    p: np.ndarray
    internal: list[tuple[int, int]]
    # direction 0: internal v => boundary c; direction 1: boundary c => internal v
    cross: list[tuple[int, int, int]]
    boundary: np.ndarray

    def __post_init__(self):
        self.x = bits(len(self.p))
        self.w = product_weights(self.x, self.p)
        for u, v in self.internal:
            self.w *= (self.x[:, u] <= self.x[:, v])
        self.compat = np.ones((len(self.x), len(self.boundary)), dtype=bool)
        for direction, v, c in self.cross:
            if direction == 0:
                self.compat &= self.x[:, v, None] <= self.boundary[None, :, c]
            else:
                self.compat &= self.boundary[None, :, c] <= self.x[:, v, None]
        self.factor = self.w @ self.compat
        self.support = self.factor > 0
        self.numerator = self.compat.T @ (self.w[:, None] * self.x)
        self.conditional = np.divide(
            self.numerator, self.factor[:, None],
            out=np.zeros_like(self.numerator), where=self.factor[:, None] > 0,
        )
        self.interface = sorted({v for _, v, _ in self.cross})

    def envelopes(self, depth):
        """Partition the interface into disjoint prefix classes.

        Production versions would obtain masses/moments from a clamped local
        solver and forced values by implication propagation. Here enumeration
        verifies that the cube envelopes and local moments have valid semantics.
        """
        selected = self.interface[:depth]
        lower = np.zeros(len(self.boundary))
        upper = np.zeros_like(lower)
        lower_numerator = np.zeros((len(self.boundary), len(self.p)))
        lower_lift = np.zeros_like(self.compat, dtype=float)
        leaves = []
        for assignment in bits(len(selected)):
            keep = self.w > 0
            if selected:
                keep &= np.all(self.x[:, selected] == assignment, axis=1)
            if not np.any(keep):
                continue
            a = self.w[keep].sum()
            moment = (self.w[keep, None] * self.x[keep]).sum(axis=0) / a
            can_zero = np.any(self.x[keep] == 0, axis=0)
            can_one = np.any(self.x[keep] == 1, axis=0)
            lo = np.ones(len(self.boundary), dtype=bool)
            hi = np.ones_like(lo)
            for direction, v, c in self.cross:
                if direction == 0:
                    # b=0 is universally allowed iff v is forced zero;
                    # b=0 is possibly allowed iff v is not forced one.
                    if can_one[v]:
                        lo &= self.boundary[:, c] == 1
                    if not can_zero[v]:
                        hi &= self.boundary[:, c] == 1
                else:
                    if can_zero[v]:
                        lo &= self.boundary[:, c] == 0
                    if not can_one[v]:
                        hi &= self.boundary[:, c] == 0
            hi &= self.support  # exact projected logical support
            lower += a * lo
            upper += a * hi
            lower_numerator += a * lo[:, None] * moment
            lower_lift[keep] = self.w[keep, None] * lo[None, :]
            leaves.append({"mass": float(a), "cube": lo, "moment": moment})
        assert np.all(lower <= self.factor + 1e-12)
        assert np.all(self.factor <= upper + 1e-12)
        assert np.all(lower_lift <= self.w[:, None] * self.compat + 1e-12)
        return lower, upper, lower_numerator, lower_lift, leaves


def make_model(seed, k=None, n=None):
    rng = np.random.default_rng(seed)
    k = k or int(rng.integers(4, 7))
    ns = [n or int(rng.integers(3, 6)) for _ in range(2)]
    boundary = bits(k)
    pc = rng.uniform(.08, .92, k)
    core = product_weights(boundary, pc)
    positions = rng.uniform(0, 1, k)
    for c in range(k - 1):
        if rng.random() < .25:
            u, v = sorted([c, c+1], key=lambda j: positions[j])
            core *= boundary[:, u] <= boundary[:, v]
    modules = []
    for size in ns:
        pm = rng.uniform(.08, .92, size)
        pos = rng.uniform(0, 1, size)
        # Internal chain makes the module connected; direction follows a DAG order.
        internal = [tuple(sorted((i, i+1), key=lambda v: pos[v])) for i in range(size-1)]
        cross = []
        for v in range(size):
            for c in range(k):
                if rng.random() < .30:
                    cross.append((0 if pos[v] < positions[c] else 1, v, c))
        if not cross:
            cross.append((0 if pos[0] < positions[0] else 1, 0, 0))
        modules.append(Module(pm, internal, cross, boundary))
    return boundary, core, modules


def global_truth(boundary, core, modules):
    m0, m1 = modules
    # Explicit original full joint, with no separator recovery formula.
    joint = (core[:, None, None]
             * (m0.w[:, None] * m0.compat).T[:, :, None]
             * (m1.w[:, None] * m1.compat).T[:, None, :])
    z = joint.sum()
    p = joint / z
    marg = np.concatenate([p.sum(axis=(1, 2)) @ boundary,
                           p.sum(axis=(0, 2)) @ m0.x,
                           p.sum(axis=(0, 1)) @ m1.x])
    return z, p, marg


def verify_envelopes():
    max_checks_error = 0.
    traces = []
    configurations = 0
    for seed in range(100):
        boundary, core, modules = make_model(seed)
        z, p, target = global_truth(boundary, core, modules)
        prev_l, prev_u = 0., math.inf
        trace = []
        max_depth = max(len(m.interface) for m in modules)
        for depth in range(max_depth + 1):
            values = [m.envelopes(depth) for m in modules]
            l0,u0,h0,lift0,_ = values[0]
            l1,u1,h1,lift1,_ = values[1]
            zl = float(np.sum(core*l0*l1))
            zu = float(np.sum(core*u0*u1))
            assert prev_l <= zl + 1e-12 and zu <= prev_u + 1e-12
            assert zl <= z + 1e-12 and z <= zu + 1e-12
            prev_l, prev_u = zl, zu
            certificate = 1. - zl/zu
            err = tv = None
            if zl > 1e-300:
                # Recover all module marginals by contracting leaf moments.
                marg = np.concatenate([
                    (core*l0*l1) @ boundary / zl,
                    (core*l1) @ h0 / zl,
                    (core*l0) @ h1 / zl,
                ])
                # Independent lifted lower joint verifies latent-moment recovery.
                jl = core[:,None,None] * lift0.T[:,:,None] * lift1.T[:,None,:]
                pl = jl / zl
                check = np.concatenate([
                    pl.sum(axis=(1,2)) @ boundary,
                    pl.sum(axis=(0,2)) @ modules[0].x,
                    pl.sum(axis=(0,1)) @ modules[1].x,
                ])
                max_checks_error = max(max_checks_error, float(np.max(abs(check-marg))))
                err = float(np.max(abs(marg-target)))
                tv = float(np.sum(abs(pl-p))/2)
                assert tv <= certificate + 2e-11
                assert err <= certificate + 2e-11
            trace.append(dict(depth=depth, leaves=[len(v[4]) for v in values],
                              Z_lower=zl, Z_upper=zu, certificate=certificate,
                              actual_max_marginal_error=err, actual_joint_TV=tv))
            configurations += 1
        assert abs(zl/z-1) < 2e-10 and abs(zu/z-1) < 2e-10
        if seed in [0, 1, 2]:
            traces.append(dict(seed=seed, exact_Z=float(z), trace=trace))
    return dict(cases=100, refinement_configurations=configurations,
                max_latent_vs_full_joint_recovery_error=max_checks_error,
                all_pointwise_bounds_monotonicity_and_TV_checks_passed=True,
                examples=traces)


def verify_sampling_recovery():
    boundary, core, modules = make_model(23, k=6, n=5)
    z, pfull, target = global_truth(boundary, core, modules)
    exact_unnorm = core * modules[0].factor * modules[1].factor
    rng = np.random.default_rng(20261014)
    # A deliberately imperfect positive separator surrogate on exact support.
    fhat = [m.factor * np.exp(1.5*rng.normal(size=len(boundary))) for m in modules]
    q_unnorm = core * fhat[0] * fhat[1]
    q = q_unnorm/q_unnorm.sum()
    h = np.concatenate([boundary, *[m.conditional for m in modules]], axis=1)
    approx_population = q @ h
    weight = np.divide(exact_unnorm,q_unnorm,out=np.zeros_like(q),where=q_unnorm>0)
    corrected_population = (q*weight) @ h / (q*weight).sum()
    assert np.max(abs(corrected_population-target)) < 1e-12
    rows = []
    for samples in [100,1000,10000]:
        errors = []
        for _ in range(30):
            s = rng.choice(len(q), size=samples, p=q)
            mu = (weight[s,None]*h[s]).sum(axis=0)/weight[s].sum()
            errors.append(float(np.mean(abs(mu-target))))
        rows.append(dict(samples=samples, replications=30,
                         median_MAE=float(np.median(errors)),
                         maximum_MAE=float(np.max(errors))))
    return dict(approximate_population_MAE=float(np.mean(abs(approx_population-target))),
                corrected_population_max_error=float(np.max(abs(corrected_population-target))),
                Monte_Carlo=rows,
                caveat="Self-normalized finite-sample marginals are consistent, not unbiased; no mixing or runtime guarantee is claimed.")


def connected_separator(m):
    """Connected, implication-only module: t=>z_i and z_i=>a_i,b_i.

    F(a,b)=2^{-(m+1)}[prod_i(1+a_i*b_i)+prod_i(a_i*b_i)].
    All Bernoulli parameters are 1/2. The full primal graph is a tree.
    """
    x = bits(2*m)
    pair = x[:,:m]*x[:,m:]
    return (np.prod(1+pair,axis=1)+np.prod(pair,axis=1))/(2.**(m+1))


def tt_ranks(values, order):
    # bits() uses little-endian coordinates; Fortran reshape preserves that.
    t = values.reshape([2]*len(order),order='F').transpose(order)
    ranks = []
    for k in range(1,len(order)):
        singular = np.linalg.svd(t.reshape(2**k,-1),compute_uv=False)
        ranks.append(int(np.sum(singular > singular[0]*1e-11)))
    return ranks


def verify_rank_obstruction():
    rows=[]
    for m in range(2,9):
        f=connected_separator(m)
        grouped=tt_ranks(f,list(range(2*m)))
        interleaved=tt_ranks(f,[j for i in range(m) for j in (i,m+i)])
        assert grouped[m-1]==2**m
        assert max(interleaved)<=3
        rows.append(dict(pairs=m,boundary_size=2*m,clamped_internal_treewidth=1,
                         full_graph_treewidth=1,grouped_max_TT_rank=max(grouped),
                         interleaved_max_TT_rank=max(interleaved)))
    return rows


def verify_wide_boundary():
    k=1082
    p=np.full(k,.999)
    child_prior=.4
    all_on=math.exp(float(np.log(p).sum()))
    z=(1-child_prior)+child_prior*all_on
    child=child_prior*all_on/z
    parents=(p*(1-child_prior)+child_prior*all_on)/z
    return dict(boundary_size=k,latent_states=2,exact_nonnegative_TT_rank=2,
                dense_boundary_entries=f"2^{k}",
                exact_Z=z,exact_child_marginal=child,
                exact_parent_marginal_min=float(parents.min()),
                exact_parent_marginal_max=float(parents.max()),
                root_certificate=1-all_on,after_one_split_certificate=0.,
                computation="Analytic contraction of two rank-one cubes; no boundary table allocated. This is a tree and also easy for standard exact inference.")


def verify_gradient_trap():
    p0=.5
    coefficient=8.
    def zhat(p):
        # exact F(0)=1-p,F(1)=1, multiplied by a weight-dependent scalar.
        return (.5*(1-p)+.5)*math.exp(coefficient*(p-p0))
    def logistic(x): return 1/(1+math.exp(-x))
    step=1e-6
    derivative=(math.log(zhat(logistic(step)))-math.log(zhat(logistic(-step))))/(2*step)
    inferred=p0+derivative
    target=1/3
    assert inferred>1
    assert abs(inferred-(target+coefficient/4))<1e-8
    return dict(separator_value_error_at_evaluation_point=0.,true_marginal=target,
                naive_differentiation_marginal=inferred,
                failure="Exact separator values at one weight vector do not imply correct weight derivatives of a fitted surrogate.")


def verify_nmae_trap():
    # Two wide-boundary assignments have very different outside/cavity weights.
    f=np.array([1e-8,1.])
    fhat=np.array([2e-8,1.])
    cavity=np.array([1e8,1.])
    posterior=cavity*f;posterior/=posterior.sum()
    approx=cavity*fhat;approx/=approx.sum()
    return dict(uniform_factor_NMAE=float(np.mean(abs(f-fhat))/np.max(f)),
                posterior_absolute_error=float(abs(posterior[0]-approx[0])),
                exact_event_probability=float(posterior[0]),
                approximate_event_probability=float(approx[0]))


def verify_uniform_query_obstruction():
    rows=[]
    for k in [24,176,1082]:
        # A totally ordered implication boundary has exactly k+1 valid assignments.
        with localcontext() as ctx:
            ctx.prec=50
            probability=Decimal(k+1)/(Decimal(2)**k)
            expected=5000*probability
        rows.append(dict(boundary_size=k,supported_assignments=k+1,
                         uniform_query_support_probability=format(probability,'.10E'),
                         expected_supported_queries_from_5000=format(expected,'.10E'),
                         probability_at_least_one_supported_query_upper_bound=format(min(Decimal(1),expected),'.10E')))
    return rows


def main():
    results=dict(
        scope="New feasibility prototypes on synthetic implication DAGs, not GO/ChEBI replications and not a novelty proof.",
        certified_cube_envelopes=verify_envelopes(),
        sampling_recovery=verify_sampling_recovery(),
        ordering_rank_obstruction=verify_rank_obstruction(),
        wide_boundary_analytic_case=verify_wide_boundary(),
        derivative_counterexample=verify_gradient_trap(),
        factor_error_counterexample=verify_nmae_trap(),
        uniform_query_obstruction=verify_uniform_query_obstruction(),
    )
    (ROOT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in results.items() if k!='certified_cube_envelopes'},ensure_ascii=False,indent=2))
    cert=results['certified_cube_envelopes']
    print('CERTIFICATE CHECKS:',cert['cases'],'models,',cert['refinement_configurations'],'configurations; max recovery discrepancy',cert['max_latent_vs_full_joint_recovery_error'])


if __name__=='__main__':
    main()
