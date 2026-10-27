"""P4.7d replication on its original 100 proteins, plus log-ratio BP and exact C2 weights."""
import sys,json,time
from pathlib import Path
import numpy as np
from scipy.special import logit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'journal_comparison'))
from p4_7_bp_audit_2026_10_22 import bp_logratios,compile_circuit
sys.path.insert(0,str(ROOT/'experiments/scripts'))
from wmc_common import load_theory,DATA_DIR
from wmc_inference import loopy_bp_theta
from consistency import isotonic_dag_hildreth
from learned_constrained import term_depth,ModelC2
from p4_3_learned_constrained_model_vs_calibration import metrics

ns=sys.argv[1]
ids,edges=load_theory('dgp2_'+ns);n=len(ids);circ=compile_circuit(n,edges)
z=np.load(DATA_DIR/'scores'/('deepgoplus2_'+ns+'.npz'))
root=int(z['root_index']);cols=np.arange(n)!=root
Y=z['labels'].astype(float);keep=Y[:,cols].sum(1)>0;Y=Y[keep]
S=np.clip(z['cnn'].astype(float)[keep],1e-6,1-1e-6)
cache=np.load(DATA_DIR/'scores'/('theta_star_'+ns+'_cnn_eps0.001_v2.npz'))
rows=np.random.default_rng(3).choice(np.flatnonzero(~np.isnan(cache['err'])),100,False)
TH=cache['theta'];depth=term_depth(n,edges,root)
iso=isotonic_dag_hildreth(S[rows],edges)[0]
target=.999*iso+.001*circ.marginals(np.full(n,.5))[0]
L=logit(S[rows]);M=logit(np.clip(target,1e-9,1-1e-9))
model=ModelC2(circ,depth)
model.w=np.array(json.load(open(ROOT/'experiments/results'/('p4_5_nested_C2_'+ns+'_cnn.json')))['C2_fit']['w']).ravel()
out=dict(ns=ns,proteins=100,stages={})
for stage in ('theta_star','C2_exact_trained_weights'):
    preds={k:[] for k in ('exact','production','logratio05','logratio01')}
    conv={k:[] for k in preds};ts={k:0. for k in preds}
    for j,row in enumerate(rows):
        th=TH[row] if stage=='theta_star' else model._theta(model.w,TH[row],L[j],M[j])
        for name,fn in [('exact',lambda:circ.marginals_theta(th)[0]),
                        ('production',lambda:loopy_bp_theta(n,edges,th,tol=1e-12,max_iter=20000)),
                        ('logratio05',lambda:bp_logratios(n,edges,th,step=.5)),
                        ('logratio01',lambda:bp_logratios(n,edges,th,step=.1))]:
            t=time.time();res=fn();ts[name]+=time.time()-t
            preds[name].append(res if name=='exact' else res['mu'])
            conv[name].append(True if name=='exact' else res['converged'])
    P={k:np.array(v) for k,v in preds.items()};truth=P['exact']
    rec={}
    for name,p in P.items():
        mx=np.abs(p-truth).max(1)
        rec[name]=dict(metrics=metrics(p[:,cols],Y[rows][:,cols]),
                       max_error_quantiles=np.quantile(mx,[.5,.9,1]).tolist(),
                       frac_error_over_point1=float(np.mean(mx>.1)),seconds=ts[name],
                       converged_fraction=float(np.mean(conv[name])))
    out['stages'][stage]=rec
    print(ns,stage,json.dumps(rec),flush=True)
if True:
    orig=json.load(open(ROOT/'experiments/results'/('p4_7d_bp_at_exact_theta_star_'+ns+'_cnn.json')))
    out['production_reproduction_max_metric_difference']=max(abs(orig['metrics']['bp_at_theta_star'][k]-out['stages']['theta_star']['production']['metrics'][k]) for k in orig['metrics']['bp_at_theta_star'])
    out['production_reproduction_max_error_quantile_difference']=float(np.max(abs(np.array(orig['max_err_quantiles'])-out['stages']['theta_star']['production']['max_error_quantiles'])))
(ROOT/'journal_comparison'/('Codex_P4.7_BP_full100_'+ns+'_2026-10-22.json')).write_text(json.dumps(out,indent=2))
