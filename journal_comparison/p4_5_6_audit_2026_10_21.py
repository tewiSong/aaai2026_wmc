"""Independent C2 audit. Read-only with respect to Claude's artifacts.

Recompute predictions, inspect clipping, and test marginal residual controls with
the same baseline and penalty scaling as the joint residual model.
Run on a Slurm compute node through experiments/run.sh.
"""
import argparse
import inspect
import json
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/scripts'))
from consistency import isotonic_dag_hildreth
from learned_constrained import ModelC2, term_depth, log_partition
from p4_3_learned_constrained_model_vs_calibration import metrics
from wmc_common import DATA_DIR, load_theory, write_dimacs, undirected_adjacency
from wmc_inference import Circuit, compile_with_d4, ve_log_z
from p0_3_minfill_width_check_vs_paper import eliminate_greedy


# Same arithmetic circuit, evaluated with stable log literal weights directly.
# This avoids changing the model by clipping sigmoid(theta).
src = textwrap.dedent(inspect.getsource(Circuit.marginals))
src = src.replace('def marginals(self, p: np.ndarray)', 'def stable_marginals(self, theta: np.ndarray)')
src = src.replace('logp, log1mp = np.log(p), np.log1p(-p)',
                  'p = expit(theta)\n    logp, log1mp = -np.logaddexp(0., -theta), -np.logaddexp(0., theta)')
scope = dict(np=np, expit=expit)
exec(src, scope)
stable_marginals = scope['stable_marginals']


def fit_marginal_residual(L, M, Y, depth, anchor, matched_penalty):
    D = int(depth.max()) + 1
    W = np.zeros((D, 3))
    fits = []
    for d in range(D):
        mask = depth == d
        if not mask.any():
            continue
        l, m, y = L[:, mask].ravel(), M[:, mask].ravel(), Y[:, mask].ravel()
        X = np.c_[np.ones_like(l), l, m]
        offset = m if anchor == 'M' else l
        ridge = 1e-3 / int(mask.sum()) if matched_penalty else 1e-3
        def f(w):
            eta = offset + X @ w
            loss = np.mean(np.logaddexp(0., eta) - y * eta) + ridge * (w @ w)
            grad = X.T @ (expit(eta) - y) / len(y) + 2 * ridge * w
            return loss, grad
        fit = minimize(f, np.zeros(3), jac=True, method='L-BFGS-B',
                       options=dict(maxiter=1000, ftol=1e-12, gtol=1e-8))
        W[d] = fit.x
        fits.append(dict(depth=d, terms=int(mask.sum()), success=bool(fit.success),
                         grad_max=float(np.abs(fit.jac).max()), ridge=ridge))
    return W, fits


def main(ns, inp):
    ids, edges = load_theory('dgp2_' + ns)
    n = len(ids)
    z = np.load(DATA_DIR / 'scores' / ('deepgoplus2_' + ns + '.npz'))
    S, Y, root = z[inp].astype(float), z['labels'].astype(float), int(z['root_index'])
    cols = np.arange(n) != root
    keep = Y[:, cols].sum(1) > 0
    S, Y = np.clip(S[keep], 1e-6, 1-1e-6), Y[keep]
    perm = np.random.default_rng(0).permutation(len(S))
    fi, ei = perm[:len(S)//2], perm[len(S)//2:]
    depth = term_depth(n, edges, root)
    with tempfile.TemporaryDirectory() as td:
        cnf, nnf = Path(td)/'t.cnf', Path(td)/'t.nnf'
        write_dimacs(cnf, n, edges)
        compile_with_d4(cnf, nnf)
        circ = Circuit(nnf, n)
    mu_unif = circ.marginals(np.full(n, .5))[0]
    iso = isotonic_dag_hildreth(S, edges)[0]
    target = .999*iso + .001*mu_unif
    L, M = logit(S), logit(np.clip(target, 1e-9, 1-1e-9))
    cache = np.load(DATA_DIR/'scores'/('theta_star_'+ns+'_'+inp+'_eps0.001.npz'))
    TH = cache['theta']
    saved = json.load(open(ROOT/'experiments/results'/('p4_5_nested_C2_'+ns+'_'+inp+'.json')))
    ablation = json.load(open(ROOT/'experiments/results'/('p4_6_ablation_'+ns+'_'+inp+'.json')))
    c2 = ModelC2(circ, depth)
    c2.w = np.array(saved['C2_fit']['w']).ravel()
    th = np.stack([c2._theta(c2.w, TH[i], L[i], M[i]) for i in ei])
    pred = c2.predict(TH[ei], L[ei], M[ei])
    w0 = c2.predict(TH[ei], L[ei], M[ei], w=np.zeros_like(c2.w))
    out = dict(ns=ns, input=inp, proteins_eval=len(ei),
               C2_metrics=metrics(pred[:,cols], Y[ei][:,cols]),
               w0_metrics=metrics(w0[:,cols],Y[ei][:,cols]))
    out['max_metric_difference_vs_p46'] = max(abs(out['C2_metrics'][k]-ablation['metrics']['C2'][k]) for k in out['C2_metrics'])
    residual = np.abs(w0-target[ei]).max(1)
    residual_nonroot=np.abs(w0[:,cols]-target[ei][:,cols]).max(1)
    out['w0_error_eval'] = dict(max=float(residual.max()), median=float(np.median(residual)),
                              over_1e_3=int((residual>1e-3).sum()), over_1e_2=int((residual>1e-2).sum()),
                              nonroot_max=float(residual_nonroot.max()),
                              nonroot_over_1e_2=int((residual_nonroot>1e-2).sum()))
    stable_pred = pred.copy()
    changed = np.flatnonzero((np.abs(th)>logit(1-1e-12)).any(1))
    for j in changed:
        stable_pred[j] = stable_marginals(circ, th[j])[0]
    out['C2_clipping'] = dict(rows=int(len(changed)), theta_min=float(th.min()), theta_max=float(th.max()),
                             max_prediction_change=float(np.abs(stable_pred-pred).max()),
                             max_prediction_change_nonroot=float(np.abs(stable_pred[:,cols]-pred[:,cols]).max()),
                             stable_metrics=metrics(stable_pred[:,cols],Y[ei][:,cols]))
    # Inspect worst moment matches under exactly the same cached parameters.
    diagnostics=[]
    for i in np.argsort(cache['err'])[-5:]:
        _, mu_old = log_partition(circ, TH[i])
        mu_new, _ = stable_marginals(circ, TH[i])
        sat=np.abs(TH[i])>logit(1-1e-12)
        diagnostics.append(dict(protein_row=int(i), old_err=float(np.abs(mu_old-target[i]).max()),
                                stable_err=float(np.abs(mu_new-target[i]).max()),
                                theta_min=float(TH[i].min()),theta_max=float(TH[i].max()),
                                clipped_objective_gradient_mismatch=float(np.max(np.abs(expit(TH[i,sat])-mu_old[sat]))) if sat.any() else 0.))
    out['worst_moment_matches'] = diagnostics
    # Two-node counterexample for the clipped log-partition and gradient.
    with tempfile.TemporaryDirectory() as td:
        cnf,nnf=Path(td)/'toy.cnf',Path(td)/'toy.nnf'
        write_dimacs(cnf,2,np.array([[1,0]]));compile_with_d4(cnf,nnf);toy=Circuit(nnf,2)
    eta=np.array([-40.,40.]);approx_A,approx_mu=log_partition(toy,eta)
    true_mu,true_logz=stable_marginals(toy,eta)
    true_A=true_logz+np.logaddexp(0.,eta).sum()
    numeric=[]
    for j in range(2):
        e=np.eye(2)[j]*1e-4
        numeric.append((log_partition(toy,eta+e)[0]-log_partition(toy,eta-e)[0])/2e-4)
    out['toy_clipping_counterexample']=dict(theta=eta.tolist(),reported_A=float(approx_A),
        true_A=float(true_A),reported_gradient=approx_mu.tolist(),finite_difference=numeric,
        true_gradient=true_mu.tolist())
    # Independent dense elimination queries for ordinary finite literal weights.
    checks=[]
    order=eliminate_greedy(undirected_adjacency(n,edges),'min-fill')['order']
    for j in (0, len(ei)//2):
        q=int(np.flatnonzero(cols)[17])
        p=np.clip(expit(th[j]),1e-12,1-1e-12)
        ref=np.exp(ve_log_z(n,edges,p,order,clamp={q:1})-ve_log_z(n,edges,p,order))
        checks.append(dict(row=int(ei[j]),term=q,reference=float(ref),circuit=float(pred[j,q]),
                           abs_error=float(abs(ref-pred[j,q]))))
    out['ve_checks'] = checks
    out['controls']={}
    for anchor, matched in [('M',False),('M',True),('L',True)]:
        W, fits = fit_marginal_residual(L[fi],M[fi],Y[fi],depth,anchor,matched)
        F=np.stack([np.ones_like(L[ei]),L[ei],M[ei]],-1)
        offset=M[ei] if anchor=='M' else L[ei]
        raw=expit(offset+np.einsum('kni,ni->kn',F,W[depth]))
        projected=isotonic_dag_hildreth(raw,edges)[0]
        key=anchor+('_matched_penalty' if matched else '_original_penalty')
        out['controls'][key]=dict(raw=metrics(raw[:,cols],Y[ei][:,cols]),
                                   isotonic=metrics(projected[:,cols],Y[ei][:,cols]),fits=fits)
        print(ns,inp,key,out['controls'][key]['isotonic'],flush=True)
    dest=ROOT/'journal_comparison'/('Codex_P4.5_6_audit_'+ns+'_'+inp+'_2026-10-21.json')
    dest.write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print('audit_saved',str(dest),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',type=int,required=True)
    a=ap.parse_args();main(*[('go_cc','cnn'),('go_cc','fused'),('go_mf','cnn'),('go_mf','fused')][a.case])
