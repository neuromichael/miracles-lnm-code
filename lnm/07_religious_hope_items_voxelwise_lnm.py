#!/usr/bin/env python3
"""
Voxelwise lesion network maps of each religious hope item (Grafman VHIS cohort),
followed by the pairwise spatial correlations between the six item maps

Items: belief_1_hope_1 to belief_6_hope_6, items 12 to 17 of the Sethi and
Seligman (1993) Religiousness Measure, each scored 1 (strongly disagree) to 7
(strongly agree), none reverse scored.

Model at every voxel v, across subjects i, run once per item:
    N_i(v) = b0(v) + b_item(v) * item_i + b_vol(v) * lesion_volume_i + e_i(v)

N_i(v) is the subject's lesion network map (<ID>_func_seed_T.nii). Lesion volume
in mm^3 (from <ID>.nii) is a nuisance covariate. A subject who skipped an item is
left out of that item's model only. The voxel mask is the same for every item
(voxels nonzero in all subjects with imaging). The output per item is the
unthresholded voxelwise t map for the item regressor.

Sign convention: positive t at voxel v means stronger lesion connectivity to v
goes with stronger agreement with the item.

Spatial correlation: Pearson r between two item t maps across in-mask voxels.

Outputs (OUTPUT_DIR):
  volumecontrolled_hope_<k>_vhis_t_map.nii.gz  (k = 1 to 6)
  religious_hope_items_spatial_correlations.xlsx
  included_subjects.csv
  run_log.txt
"""
import os, time
import numpy as np, pandas as pd, nibabel as nib

IMAGE_DIR   = os.environ["VHIS_IMAGE_DIR"]          # <ID>.nii lesion masks and <ID>_func_seed_T.nii network maps
DATA_DIR    = os.environ["VHIS_DATA_DIR"]           # behavioral spreadsheets (not public)
OUTPUT_ROOT = os.environ.get("VHIS_OUTPUT_DIR", "outputs")
SPREADSHEET = os.path.join(DATA_DIR, "Dataset_Warfighter_Religious.xlsx")
OUTPUT_DIR  = os.path.join(OUTPUT_ROOT, "religious_hope_items_voxelwise_lnm")
ITEMS = {f"hope_{k}": f"belief_{k}_hope_{k}" for k in range(1, 7)}
TEXT  = {"hope_1": "There is a heaven?",
         "hope_2": "It is possible for all humans to live in harmony together?",
         "hope_3": "There really are miracles?",
         "hope_4": "Your suffering will be rewarded?",
         "hope_5": "In the future your children will be able to lead a better life than yourself?",
         "hope_6": "The future will be a better place to live?"}

os.makedirs(OUTPUT_DIR, exist_ok=True)
logf = open(os.path.join(OUTPUT_DIR, "run_log.txt"), "w")
def log(m):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True); logf.write(line + "\n"); logf.flush()

log(f"script: {os.path.abspath(__file__)}")
log(f"images: {IMAGE_DIR}"); log(f"scores: {SPREADSHEET}")

raw = pd.read_excel(SPREADSHEET, sheet_name="data")
df = raw[["id_nih"] + list(ITEMS.values())].rename(columns={v: k for k, v in ITEMS.items()})
df = df.dropna(subset=["id_nih"]); df["VHIS_ID"] = df["id_nih"].astype(int)
df["diagnosis"] = np.where(df.VHIS_ID < 4000, 1, 2)

rows = []
for r in df.itertuples():
    net = os.path.join(IMAGE_DIR, f"{r.VHIS_ID:04d}_func_seed_T.nii")
    les = os.path.join(IMAGE_DIR, f"{r.VHIS_ID:04d}.nii")
    if os.path.exists(net) and os.path.exists(les):
        rows.append(dict(VHIS_ID=r.VHIS_ID, diagnosis=int(r.diagnosis),
                         **{k: getattr(r, k) for k in ITEMS}, network_file=net, lesion_file=les))
sub = pd.DataFrame(rows)
log(f"subjects with network + lesion: {len(sub)} of {len(df)}")

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

mask = np.all(np.isfinite(D), axis=0) & np.all(D != 0, axis=0)
mask &= D.var(axis=0) > 0
Yall = D[:, mask].astype(np.float64); del D
log(f"in-mask voxels (nonzero in every subject): {int(mask.sum())}")

tmaps, info = {}, []
for it in ITEMS:
    ok = sub[it].notna().values
    x = sub.loc[ok, it].values.astype(float); Y = Yall[ok]; m = int(ok.sum())
    M = np.column_stack([x, np.ones(m), vol[ok]])
    dof = m - M.shape[1]
    B = np.linalg.pinv(M) @ Y
    R = Y - M @ B
    s2 = (R * R).sum(axis=0) / dof
    cvar = np.linalg.inv(M.T @ M)[0, 0]
    t = B[0] / np.sqrt(s2 * cvar)
    tmaps[it] = t
    full = np.zeros(nv, dtype=np.float32); full[mask] = t
    img = nib.Nifti1Image(full.reshape(shape), aff)
    img.header["descrip"] = f"VHIS {it} voxelwise t, lesion vol covaried".encode()[:79]
    out = os.path.join(OUTPUT_DIR, f"volumecontrolled_{it}_vhis_t_map.nii.gz")
    nib.save(img, out)
    info.append(dict(item=it, item_text=TEXT[it], n_subjects=m, dof=dof, mean=x.mean(), sd=x.std(ddof=1),
                     t_min=t.min(), t_max=t.max(), file=os.path.basename(out)))
    log(f"{it}: n {m}, dof {dof}, mean {x.mean():.3f} sd {x.std(ddof=1):.3f}, t range {t.min():.3f} to {t.max():.3f}; saved {out}")

names = list(ITEMS)
Rmat = pd.DataFrame(np.corrcoef(np.vstack([tmaps[k] for k in names])), index=names, columns=names)
pairs = pd.DataFrame([dict(map_a=a, map_b=b, pearson_r=Rmat.loc[a, b], n_voxels=int(mask.sum()))
                      for i, a in enumerate(names) for b in names[i + 1:]])
log("pairwise spatial correlations (Pearson r):\n" + Rmat.round(3).to_string())
xl = os.path.join(OUTPUT_DIR, "religious_hope_items_spatial_correlations.xlsx")
with pd.ExcelWriter(xl) as w:
    Rmat.to_excel(w, sheet_name="correlation_matrix")
    pairs.to_excel(w, sheet_name="pairs", index=False)
    pd.DataFrame(info).to_excel(w, sheet_name="maps", index=False)
log(f"saved {xl}"); logf.close()
