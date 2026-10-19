#!/usr/bin/env python3
"""
Voxelwise lesion network map of the religious hope subscale (Grafman VHIS cohort)

Religious hope subscale: items 12 to 17 of the Sethi and Seligman (1993)
Religiousness Measure, columns belief_1_hope_1 to belief_6_hope_6, each scored
1 (strongly disagree) to 7 (strongly agree), none reverse scored. The subscale
score is the sum of the six items. Where a subject skipped items, the sum is
prorated (mean of the answered items times 6) so that no subject is dropped.

Model at every voxel v, across subjects i:
    N_i(v) = b0(v) + b_hope(v) * hope_i + b_vol(v) * lesion_volume_i + e_i(v)

N_i(v) is the subject's lesion network map (normative connectome t statistic of
lesion-to-voxel functional connectivity, file <ID>_func_seed_T.nii). Lesion
volume in mm^3 (from <ID>.nii) is a nuisance covariate, matching the
"volumecontrolled" miracles and magical ideation maps. The output is the
unthresholded voxelwise t map for the hope regressor.

Sign convention: positive t at voxel v means stronger lesion connectivity to v
goes with a higher religious hope score.

Outputs (OUTPUT_DIR):
  volumecontrolled_religious_hope_vhis_t_map.nii.gz
  included_subjects.csv
  run_log.txt
"""
import os, time
import numpy as np, pandas as pd, nibabel as nib

IMAGE_DIR   = os.environ["VHIS_IMAGE_DIR"]          # <ID>.nii lesion masks and <ID>_func_seed_T.nii network maps
DATA_DIR    = os.environ["VHIS_DATA_DIR"]           # behavioral spreadsheets (not public)
OUTPUT_ROOT = os.environ.get("VHIS_OUTPUT_DIR", "outputs")
SPREADSHEET = os.path.join(DATA_DIR, "Dataset_Warfighter_Religious.xlsx")
OUTPUT_DIR  = os.path.join(OUTPUT_ROOT, "religious_hope_voxelwise_lnm")
OUT_NAME    = "volumecontrolled_religious_hope_vhis_t_map.nii.gz"
HOPE_ITEMS  = [f"belief_{k}_hope_{k}" for k in range(1, 7)]

os.makedirs(OUTPUT_DIR, exist_ok=True)
logf = open(os.path.join(OUTPUT_DIR, "run_log.txt"), "w")
def log(m):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True); logf.write(line + "\n"); logf.flush()

log(f"script: {os.path.abspath(__file__)}")
log(f"images: {IMAGE_DIR}"); log(f"scores: {SPREADSHEET}")

raw = pd.read_excel(SPREADSHEET, sheet_name="data")
df = raw[["id_nih"] + HOPE_ITEMS].copy()
df["diagnosis"] = np.where(df.id_nih < 4000, 1, 2)
df["n_items_answered"] = df[HOPE_ITEMS].notna().sum(axis=1)
df["hope"] = df[HOPE_ITEMS].mean(axis=1) * len(HOPE_ITEMS)
df = df.dropna(subset=["id_nih", "hope"]); df["VHIS_ID"] = df["id_nih"].astype(int)
log(f"scores in spreadsheet: {len(df)} (diagnosis counts {df.diagnosis.value_counts().to_dict()})")

rows = []
for r in df.itertuples():
    net = os.path.join(IMAGE_DIR, f"{r.VHIS_ID:04d}_func_seed_T.nii")
    les = os.path.join(IMAGE_DIR, f"{r.VHIS_ID:04d}.nii")
    if os.path.exists(net) and os.path.exists(les):
        rows.append(dict(VHIS_ID=r.VHIS_ID, diagnosis=int(r.diagnosis), hope=float(r.hope),
                         n_items_answered=int(r.n_items_answered),
                         network_file=net, lesion_file=les))
sub = pd.DataFrame(rows)
log(f"subjects with score + network + lesion: {len(sub)}")
log(f"no network file for {len(df) - len(sub)} IDs: {sorted(set(df.VHIS_ID) - set(sub.VHIS_ID))}")
pr = sub[sub.n_items_answered < len(HOPE_ITEMS)]
log(f"included subjects with a prorated score: {len(pr)} {dict(zip(pr.VHIS_ID, pr.n_items_answered))}")

ref = nib.load(sub.network_file.iloc[0]); shape = ref.shape[:3]; aff = ref.affine
nv = int(np.prod(shape)); n = len(sub)
D = np.zeros((n, nv), dtype=np.float32); vol = np.zeros(n)
for i, r in enumerate(sub.itertuples()):
    im = nib.load(r.network_file)
    assert im.shape[:3] == shape and np.allclose(im.affine, aff), r.network_file
    D[i] = im.get_fdata(dtype=np.float32).reshape(-1)
    L = nib.load(r.lesion_file); assert L.shape[:3] == shape
    vol[i] = (L.get_fdata() > 0).sum() * float(abs(np.linalg.det(L.affine[:3, :3])))
sub["lesion_volume_mm3"] = vol
sub.to_csv(os.path.join(OUTPUT_DIR, "included_subjects.csv"), index=False)
log(f"hope: mean {sub.hope.mean():.3f} sd {sub.hope.std():.3f} range {sub.hope.min():.1f}-{sub.hope.max():.1f}")
log(f"lesion volume mm3: mean {vol.mean():.0f} sd {vol.std():.0f}; r(hope, volume) = {np.corrcoef(sub.hope, vol)[0,1]:.3f}")

mask = np.all(np.isfinite(D), axis=0) & np.all(D != 0, axis=0)
mask &= D.var(axis=0) > 0
Y = D[:, mask].astype(np.float64)
log(f"in-mask voxels (nonzero in every subject): {int(mask.sum())}")

x = sub.hope.values.astype(float)
M = np.column_stack([x, np.ones(n), vol])
dof = n - M.shape[1]
B = np.linalg.pinv(M) @ Y
R = Y - M @ B
s2 = (R * R).sum(axis=0) / dof
cvar = np.linalg.inv(M.T @ M)[0, 0]
t = B[0] / np.sqrt(s2 * cvar)
log(f"dof {dof}; t range {t.min():.3f} to {t.max():.3f}")

full = np.zeros(nv, dtype=np.float32); full[mask] = t
img = nib.Nifti1Image(full.reshape(shape), aff)
img.header["descrip"] = b"VHIS religious hope voxelwise t, lesion volume covaried"
nib.save(img, os.path.join(OUTPUT_DIR, OUT_NAME))
log(f"saved {os.path.join(OUTPUT_DIR, OUT_NAME)}"); logf.close()
