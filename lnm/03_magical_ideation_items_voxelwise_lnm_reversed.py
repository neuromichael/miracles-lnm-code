#!/usr/bin/env python3
"""
Voxelwise lesion network maps for each of the 17 Magical Ideation items
(Grafman VHIS cohort), with items 2, 7, 8 and 14 REVERSE SCORED.

Items are coded 1 (no) / 2 (yes). Reverse scoring maps 1 -> 2 and 2 -> 1,
so that a higher value on every item indicates more magical ideation, as in
the Eckblad and Chapman (1983) scale key.

For item k, at every voxel v, across subjects i:
    N_i(v) = b0(v) + b_k(v) * item_k_i + b_vol(v) * lesion_volume_i + e_i(v)

Sign convention: positive t at voxel v means stronger lesion connectivity to v
goes with more magical ideation on that item.
"""
import os, time
import numpy as np, pandas as pd, nibabel as nib

IMAGE_DIR   = os.environ["VHIS_IMAGE_DIR"]          # <ID>.nii lesion masks and <ID>_func_seed_T.nii network maps
DATA_DIR    = os.environ["VHIS_DATA_DIR"]           # behavioral spreadsheets (not public)
OUTPUT_ROOT = os.environ.get("VHIS_OUTPUT_DIR", "outputs")
SPREADSHEET = os.path.join(DATA_DIR, "magical_ideation.xlsx")
OUTPUT_DIR  = os.path.join(OUTPUT_ROOT, "magical_ideation_voxelwise_lnm_reversed")
ITEMS = [f"magic_{k}" for k in range(1, 18)]
REVERSED = ["magic_2", "magic_7", "magic_8", "magic_14"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
logf = open(os.path.join(OUTPUT_DIR, "run_log.txt"), "w")
def log(m):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True); logf.write(line + "\n"); logf.flush()

log(f"script: {os.path.abspath(__file__)}")
log(f"images: {IMAGE_DIR}"); log(f"scores: {SPREADSHEET}")
log(f"reverse scored items: {REVERSED}")

df = pd.read_excel(SPREADSHEET)
df = df.dropna(subset=["id_nih"]); df["id_nih"] = df["id_nih"].astype(int)
for it in REVERSED:
    assert set(df[it].dropna().unique()) <= {1, 2}, it
    df[it] = 3 - df[it]
log(f"rows in spreadsheet: {len(df)} (diagnosis counts {df.diagnosis.value_counts().to_dict()})")

rows = []
for r in df.itertuples():
    net = os.path.join(IMAGE_DIR, f"{r.id_nih:04d}_func_seed_T.nii")
    les = os.path.join(IMAGE_DIR, f"{r.id_nih:04d}.nii")
    if os.path.exists(net) and os.path.exists(les):
        d = dict(id_nih=r.id_nih, diagnosis=int(r.diagnosis), network_file=net, lesion_file=les)
        d.update({it: getattr(r, it) for it in ITEMS})
        rows.append(d)
sub = pd.DataFrame(rows)
log(f"subjects with network + lesion: {len(sub)}")

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
sub["magical_ideation_total"] = sub[ITEMS].sum(axis=1) - len(ITEMS)   # number of items endorsed after reversal
sub.to_csv(os.path.join(OUTPUT_DIR, "included_subjects_reversed.csv"), index=False)
log(f"lesion volume mm3: mean {vol.mean():.0f} sd {vol.std():.0f}")

mask = np.all(np.isfinite(D), axis=0) & np.all(D != 0, axis=0)
mask &= D.var(axis=0) > 0
Yall = D[:, mask].astype(np.float64)
log(f"in-mask voxels (nonzero in every subject): {int(mask.sum())}")

for it in ITEMS:
    keep = sub[it].notna().values
    x = sub.loc[keep, it].values.astype(float)
    Y = Yall[keep]; v = vol[keep]; nk = int(keep.sum())
    M = np.column_stack([x, np.ones(nk), v])
    dof = nk - M.shape[1]
    B = np.linalg.pinv(M) @ Y
    R = Y - M @ B
    s2 = (R * R).sum(axis=0) / dof
    cvar = np.linalg.inv(M.T @ M)[0, 0]
    t = B[0] / np.sqrt(s2 * cvar)
    full = np.zeros(nv, dtype=np.float32); full[mask] = t
    img = nib.Nifti1Image(full.reshape(shape), aff)
    tag = " (reverse scored)" if it in REVERSED else ""
    img.header["descrip"] = f"VHIS {it}{tag} voxelwise t, vol covaried".encode()[:80]
    out = os.path.join(OUTPUT_DIR, f"volumecontrolled_{it}_vhis_t_map.nii.gz")
    nib.save(img, out)
    log(f"{it}{tag}: n {nk}, mean {x.mean():.3f}, dof {dof}, t range {t.min():.3f} to {t.max():.3f} -> {os.path.basename(out)}")
log("done"); logf.close()
