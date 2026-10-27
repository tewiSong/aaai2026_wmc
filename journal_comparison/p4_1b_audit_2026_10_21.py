"""Independent audit of P4.1b. Does not modify production experiment files.

Checks normalized inputs, reproduces cc/mf point estimates and paired Fmax
intervals, verifies isotonic projections against a generic constrained solver,
and checks BP-scale inputs/cheap metrics without rerunning the full BP job.
"""
from pathlib import Path
import hashlib
import json
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
from consistency import isotonic_dag_hildreth
from p4_0_downstream_conditioning_deepgoplus_cc_mf import fmax, max_closure, ece
from p4_1b_downstream_consistency_methods_deepgoplus import micro_ap
from wmc_common import load_theory, write_dimacs, undirected_adjacency
from wmc_inference import Circuit, compile_with_d4, ve_log_z
from p0_3_minfill_width_check_vs_paper import eliminate_greedy
sys.path.insert(0, str(ROOT / "journal_comparison"))
from p4_0_audit_2026_10_20 import boot_fmax


def ms(P, Y):
    return dict(fmax=float(fmax(P, Y)), micro_ap=micro_ap(P, Y),
                brier=float(np.mean((P-Y)**2)), ece=float(ece(P,Y)))


def main():
    out = dict(date="2026-10-21", author="OpenAI Codex", cases={})
    # Independent small-graph QP checks, distinct random seed from production tests.
    rng = np.random.default_rng(20261021)
    differences = []
    for _ in range(24):
        n = 12
        edges = np.array([(i,j) for i in range(n) for j in range(i)
                          if rng.random()<.25], dtype=np.int32).reshape(-1,2)
        p = rng.uniform(.001,.999,n)
        mu, info = isotonic_dag_hildreth(p[None], edges, tol=1e-12)
        constraints = [{"type":"ineq","fun":lambda x,c=c,q=q:x[q]-x[c]}
                       for c,q in edges]
        ref = minimize(lambda x:.5*np.sum((x-p)**2),p,jac=lambda x:x-p,
                       constraints=constraints, method="SLSQP",
                       options=dict(ftol=1e-14,maxiter=2000))
        assert ref.success, ref.message
        differences.append(float(np.max(np.abs(mu[0]-ref.x))))
    out["independent_qp_max_abs_error"] = max(differences)
    assert max(differences)<1e-7

    for ns in ["go_cc","go_mf","go_bp"]:
        data_path=ROOT/f"experiments/data/scores/deepgoplus2_{ns}.npz"
        z=np.load(data_path)
        ids,edges=load_theory("dgp2_"+ns)
        root=int(z["root_index"])
        cols=np.arange(len(ids))!=root
        Y=z["labels"]
        assert np.array_equal(ids,z["vocab_ids"])
        assert np.sum(Y[:,edges[:,0]]>Y[:,edges[:,1]])==0
        keep=Y[:,cols].sum(1)>0
        Y=Y[keep]
        Yc=Y[:,cols]
        circuit=None
        if ns!="go_bp":
            order=eliminate_greedy(undirected_adjacency(len(ids),edges),"min-fill")["order"]
        with tempfile.TemporaryDirectory() as td:
            if ns!="go_bp":
                cnf,nnf=Path(td)/"t.cnf",Path(td)/"t.nnf"
                write_dimacs(cnf,len(ids),edges)
                compile_with_d4(cnf,nnf,timeout_s=60)
                circuit=Circuit(nnf,len(ids))
            for key in ["cnn","fused"]:
                S=z[key][keep].astype(float)
                saved=json.load(open(ROOT/f"experiments/results/p4_1b_downstream_{ns}_{key}.json"))
                outputs={"raw":S,"max_closure":max_closure(S,edges,len(ids))}
                rec=dict(proteins=len(S), terms_evaluated=int(cols.sum()),
                         input_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest())
                if circuit is not None:
                    iso,info=isotonic_dag_hildreth(S,edges)
                    outputs["isotonic"]=iso
                    outputs["exact_wmc"]=np.stack([circuit.marginals(s)[0] for s in S])
                    rec["isotonic_info"]=info
                    rec["isotonic_range"]=[float(iso.min()),float(iso.max())]
                    rec["isotonic_micro_ap_round8"]=micro_ap(np.round(iso[:,cols],8),Yc)
                    rec["isotonic_micro_ap_round6"]=micro_ap(np.round(iso[:,cols],6),Yc)
                    rec["full_vector_brier_raw_iso"]=[float(np.mean((a-Y)**2)) for a in [S,iso]]
                    # Check the strongest true-label suppression against independent VE.
                    suppression=np.where(Y==1,S-outputs["exact_wmc"],-np.inf)
                    ri,ti=np.unravel_index(np.argmax(suppression),suppression.shape)
                    lz=ve_log_z(len(ids),edges,S[ri],order)
                    ref=np.exp(ve_log_z(len(ids),edges,S[ri],order,clamp={int(ti):1})-lz)
                    rec["ve_check"]=dict(protein=str(z["proteins"][keep][ri]),term=str(ids[ti]),
                                          ve=float(ref),circuit=float(outputs["exact_wmc"][ri,ti]),
                                          abs_error=float(abs(ref-outputs["exact_wmc"][ri,ti])))
                else:
                    # Bounded audit only; no full 1000-bootstrap/BP rerun on GO bp.
                    ix=np.random.default_rng(0).choice(len(S),64,replace=False)
                    iso,info=isotonic_dag_hildreth(S[ix],edges)
                    rec["sample64_isotonic_info"]=info
                    rec["bp_exact_reference_available"]=False
                calculated={k:ms(v[:,cols],Yc) for k,v in outputs.items()}
                rec["recomputed_metrics"]=calculated
                rec["metric_abs_differences"]={k:{m:abs(v-saved["metrics"][k][m])
                                                        for m,v in mm.items()}
                                                   for k,mm in calculated.items()}
                if circuit is not None:
                    rng=np.random.default_rng(0)
                    counts=np.zeros((1000,len(S)))
                    for b in range(1000):
                        ix=rng.integers(0,len(S),len(S))
                        counts[b]=np.bincount(ix,minlength=len(S))
                    boot={k:boot_fmax(v[:,cols],Yc,counts) for k,v in outputs.items()}
                    cis={k:np.quantile(v-boot["raw"],[.025,.5,.975]).tolist()
                         for k,v in boot.items() if k!="raw"}
                    rec["paired_fmax_95ci_vs_raw"]=cis
                    rec["ci_max_abs_difference_from_saved"]={
                        k:float(np.max(np.abs(np.array(v)-saved["method_minus_ref_95ci"]["raw"][k]["fmax"])))
                        for k,v in cis.items()}
                out["cases"][ns+"_"+key]=rec
                print(ns,key,json.dumps(rec),flush=True)
    out["source_hashes"]={f:hashlib.sha256((ROOT/"experiments/scripts"/f).read_bytes()).hexdigest()
                           for f in ["consistency.py","p4_1a_build_deepgoplus_v2_inputs.py",
                                     "p4_1b_downstream_consistency_methods_deepgoplus.py"]}
    target=ROOT/"journal_comparison/Codex_P4.1b_audit_2026-10-21.json"
    target.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
    print("Saved",target,flush=True)


if __name__=="__main__":
    main()
