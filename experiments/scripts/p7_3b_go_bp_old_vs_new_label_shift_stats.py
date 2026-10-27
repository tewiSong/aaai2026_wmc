"""P7.3b  Old vs new GO bp proteins: score / label statistics (login node, exploratory)."""
import sys, json, numpy as np
sys.path.insert(0, "scripts")
from wmc_common import DATA_DIR, load_theory
from learned_constrained import term_depth
from p4_1b_downstream_consistency_methods_deepgoplus import micro_ap
ids, edges = load_theory("dgp2_go_bp"); n=len(ids)
old=np.load(DATA_DIR/"scores"/"deepgoplus2_go_bp.npz"); new=np.load(DATA_DIR/"scores"/"p7_scores_go_bp.npz")
root=int(old["root_index"]); cols=np.array([j for j in range(n) if j!=root])
Y0=old["labels"].astype(np.int8); keep=Y0[:,cols].sum(1)>0; Y0=Y0[keep][:,cols]; Y1=new["labels"].astype(np.int8)[:,cols]
depth=term_depth(n,edges,root)[cols]
q=lambda a:np.round(np.quantile(a,[.1,.5,.9]),3).tolist()
for inp in ["cnn","fused"]:
    S0=np.clip(old[inp].astype(float),1e-6,1-1e-6)[keep][:,cols]; S1=np.clip(new[inp].astype(float),1e-6,1-1e-6)[:,cols]
    print("==",inp)
    for nm,S,Y in (("old",S0,Y0),("new",S1,Y1)):
        print(nm, "maxscore",q(S.max(1)),"n>0.1",q((S>0.1).sum(1)),"n>0.01",q((S>0.01).sum(1)),"labels",q(Y.sum(1)),
              "pred/label mass %.3f"%(S.sum()/Y.sum()), "AP raw %.4f"%micro_ap(S,Y),
              "label depth mean %.2f"%((Y*depth).sum()/Y.sum()), "posfrac S<0.01 %.3f"%(Y[S<0.01].sum()/Y.sum()))
    # term frequency shift
    f0,f1=Y0.mean(0),Y1.mean(0)
    print("term-freq corr old/new %.3f"%np.corrcoef(f0,f1)[0,1], "terms pos in new but 0 in old:",int(((f1>0)&(f0==0)).sum()))
