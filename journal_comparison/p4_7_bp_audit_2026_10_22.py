"""Audit BP at C2's exact parameters, including stopping criteria and equivalent theories."""
import argparse,json,sys,tempfile,time
from pathlib import Path
import numpy as np
import networkx as nx
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/scripts'))
from wmc_common import load_theory,write_dimacs,DATA_DIR
from wmc_inference import Circuit,compile_with_d4,loopy_bp_theta


def bp_logratios(n,edges,theta,step=.5,tol=1e-10,maxiter=30000):
    """Same sum-product fixed-point equations; damping and stopping in log ratios."""
    c,p=edges[:,0],edges[:,1]
    up=np.zeros(len(edges));dn=up.copy()
    for it in range(1,maxiter+1):
        h=theta.copy();np.add.at(h,p,up);np.add.at(h,c,dn)
        nu=np.logaddexp(0.,h[c]-dn)
        nd=-np.logaddexp(0.,-(h[p]-up))
        delta=max(np.max(np.abs(nu-up)),np.max(np.abs(nd-dn)))
        up+=step*(nu-up);dn+=step*(nd-dn)
        if delta<tol:break
    h=theta.copy();np.add.at(h,p,up);np.add.at(h,c,dn)
    return dict(mu=expit(h),converged=bool(delta<tol),iterations=it,delta=float(delta))


def compile_circuit(n,edges):
    with tempfile.TemporaryDirectory() as td:
        a,b=Path(td)/'t.cnf',Path(td)/'t.nnf';write_dimacs(a,n,edges);compile_with_d4(a,b)
        return Circuit(b,n)


def main(ns):
    toy_edges=np.array([[1,0]],dtype=np.int32);toytheta=np.array([-40.,40.]);toy=compile_circuit(2,toy_edges)
    truth=toy.marginals_theta(toytheta)[0]
    tests={}
    for name,fn in [('production_1e-10',lambda:loopy_bp_theta(2,toy_edges,toytheta,tol=1e-10,max_iter=5000)),
                    ('production_1e-13',lambda:loopy_bp_theta(2,toy_edges,toytheta,tol=1e-13,max_iter=20000)),
                    ('production_no_early_stop',lambda:loopy_bp_theta(2,toy_edges,toytheta,tol=0.,max_iter=200)),
                    ('logratio',lambda:bp_logratios(2,toy_edges,toytheta))]:
        r=fn();tests[name]=dict(mu=r['mu'].tolist(),err=float(abs(r['mu']-truth).max()),
                                iterations=r['iterations'],converged=r['converged'])
    # Independent enumeration verifies the new update on strongly weighted trees.
    rng=np.random.default_rng(981);tree_err=[]
    for k in range(8):
        n=8;e=np.array([(j,int(rng.integers(j))) for j in range(1,n)],dtype=np.int32)
        th=rng.normal(0,20,n);c=compile_circuit(n,e)
        r=bp_logratios(n,e,th)
        tree_err.append(float(abs(c.marginals_theta(th)[0]-r['mu']).max()))
    assert max(tree_err)<1e-8,tree_err
    ids,edges=load_theory('dgp2_'+ns);n=len(ids)
    circ=compile_circuit(n,edges)
    g=nx.DiGraph();g.add_nodes_from(range(n));g.add_edges_from(map(tuple,edges.tolist()))
    reduced=np.array(list(nx.transitive_reduction(g).edges()),dtype=np.int32)
    cache=np.load(DATA_DIR/'scores'/('theta_star_'+ns+'_cnn_eps0.001_v2.npz'))
    rows=np.random.default_rng(3).choice(np.flatnonzero(~np.isnan(cache['err'])),100,False)[:16]
    records=[]
    methods=[('production_1e-12',edges,lambda e,th:loopy_bp_theta(n,e,th,tol=1e-12,max_iter=20000)),
             ('production_1e-15',edges,lambda e,th:loopy_bp_theta(n,e,th,tol=1e-15,max_iter=50000)),
             ('logratio_step05',edges,lambda e,th:bp_logratios(n,e,th,step=.5)),
             ('logratio_step01',edges,lambda e,th:bp_logratios(n,e,th,step=.1)),
             ('reduced_production',reduced,lambda e,th:loopy_bp_theta(n,e,th,tol=1e-15,max_iter=50000)),
             ('reduced_logratio',reduced,lambda e,th:bp_logratios(n,e,th,step=.5))]
    redcirc=compile_circuit(n,reduced)
    max_reduced_difference=0.
    for row in rows:
        th=cache['theta'][row];truth=circ.marginals_theta(th)[0]
        max_reduced_difference=max(max_reduced_difference,float(abs(truth-redcirc.marginals_theta(th)[0]).max()))
        rr=dict(row=int(row),methods={})
        for name,e,fn in methods:
            t=time.time();r=fn(e,th)
            rr['methods'][name]=dict(error=float(abs(r['mu']-truth).max()),mae=float(abs(r['mu']-truth).mean()),
                                     converged=r['converged'],iterations=r['iterations'],seconds=time.time()-t)
        records.append(rr)
        print(ns,'row',row,{k:round(v['error'],5) for k,v in rr['methods'].items()},flush=True)
    summary={}
    for name,_,_ in methods:
        r=[x['methods'][name] for x in records]
        summary[name]=dict(median_error=float(np.median([x['error'] for x in r])),
                           max_error=float(max(x['error'] for x in r)),
                           converged_fraction=float(np.mean([x['converged'] for x in r])),
                           seconds=float(sum(x['seconds'] for x in r)))
    out=dict(ns=ns,proteins=len(rows),edges=len(edges),reduced_edges=len(reduced),
             toy_exact=truth.tolist() if False else [.5,.5],toy=tests,tree_max_error=max(tree_err),
             max_exact_difference_after_transitive_reduction=max_reduced_difference,summary=summary,records=records)
    dest=ROOT/'journal_comparison'/('Codex_P4.7_BP_audit_'+ns+'_2026-10-22.json')
    dest.write_text(json.dumps(out,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ns',required=True);a=p.parse_args();main(a.ns)
