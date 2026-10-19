#!/usr/bin/env python3
"""Miracles map vs 17 magical ideation item maps (items 2, 7, 8, 14 reverse scored).
Pearson r over voxels nonzero in both maps; one-sample t of the 17 r values vs 0;
Cronbach's alpha of the 17 items after reversal (106 imaging subjects)."""
import os, numpy as np, pandas as pd, nibabel as nib
from scipy import stats
BASE=os.environ.get("VHIS_OUTPUT_DIR","outputs")
MIR=os.path.join(BASE,"miracles_voxelwise_lnm","volumecontrolled_miracle_belief_vhis_t_map.nii.gz")
ITEMDIR=os.path.join(BASE,"magical_ideation_voxelwise_lnm_reversed")
REVERSED=["magic_2","magic_7","magic_8","magic_14"]
m=nib.load(MIR).get_fdata().ravel()
rows=[]
for k in range(1,18):
    it=f"magic_{k}"; x=nib.load(os.path.join(ITEMDIR,f"volumecontrolled_{it}_vhis_t_map.nii.gz")).get_fdata().ravel()
    ok=(m!=0)&(x!=0)&np.isfinite(m)&np.isfinite(x)
    r=float(np.corrcoef(m[ok],x[ok])[0,1])
    rows.append({"Item":it,"Reverse scored":"yes" if it in REVERSED else "","Pearson r":r,"Voxels":int(ok.sum())})
C=pd.DataFrame(rows); rs=C["Pearson r"].values
t,p=stats.ttest_1samp(rs,0); sem=rs.std(ddof=1)/np.sqrt(len(rs)); ci=stats.t.ppf(0.975,len(rs)-1)*sem
sub=pd.read_csv(os.path.join(ITEMDIR,"included_subjects_reversed.csv"))
X=sub[[f"magic_{k}" for k in range(1,18)]].values.astype(float); k=X.shape[1]
alpha=k/(k-1)*(1-X.var(axis=0,ddof=1).sum()/X.sum(axis=1).var(ddof=1))
summary=[("Belief in miracles vs. magical ideation items: voxelwise lesion network maps (Grafman VHIS), items 2, 7, 8, 14 reverse scored",None),(None,None),
("Maps",None),
("Miracles map","Voxelwise t map of miracles score (1 to 7, 'There really are miracles?') regressed on lesion network connectivity, lesion volume covaried"),
("Miracles map subjects",117),
("Item maps","One voxelwise t map per magical ideation item (coded 1 or 2), items 2, 7, 8, 14 reversed (1 to 2, 2 to 1), same model, lesion volume covaried"),
("Item map subjects",int(len(sub))),("Voxels compared",int(C.Voxels.iloc[0])),(None,None),
("Spatial correlations of miracles map with 17 item maps (Pearson r)",None),
("Number of items",len(rs)),("Mean r",rs.mean()),("SD",rs.std(ddof=1)),("Median",float(np.median(rs))),("Minimum",rs.min()),("Maximum",rs.max()),("SEM",sem),
("95% CI lower",rs.mean()-ci),("95% CI upper",rs.mean()+ci),("One-sample t-test vs 0: t",t),("df",len(rs)-1),("p (two-sided)",p),(None,None),
("Internal consistency of magical ideation items (after reversal)",None),("Cronbach's alpha",alpha),("Subjects",int(len(sub)))]
out=os.path.join(BASE,"miracles_vs_magical_ideation_summary_reversed.xlsx")
with pd.ExcelWriter(out) as w:
    pd.DataFrame(summary).to_excel(w,sheet_name="Summary",index=False,header=False)
    C.to_excel(w,sheet_name="Spatial correlations",index=False)
    pd.DataFrame([{"n_items":len(rs),"mean_r":rs.mean(),"sd_r":rs.std(ddof=1),"median_r":np.median(rs),"min_r":rs.min(),"max_r":rs.max(),"sem_r":sem,"ci95_low":rs.mean()-ci,"ci95_high":rs.mean()+ci,"t":t,"df":len(rs)-1,"p_two_sided":p,"cronbach_alpha":alpha}]).to_excel(w,sheet_name="descriptives_ttest",index=False)
print(C.round(4).to_string(index=False)); print(f"mean r {rs.mean():.4f} sd {rs.std(ddof=1):.4f} median {np.median(rs):.4f} t {t:.4f} p {p:.4g} CI {rs.mean()-ci:.4f} {rs.mean()+ci:.4f} alpha {alpha:.4f}"); print(out)
