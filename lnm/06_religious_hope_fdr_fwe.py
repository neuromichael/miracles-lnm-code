#!/usr/bin/env python3
"""
FDR and FWE thresholded maps for the religious hope voxelwise lesion network map

Same subjects, scores, mask, and model as run_religious_hope_voxelwise_lnm.py
(read from included_subjects.csv):
    N_i(v) = b0(v) + b_hope(v) * hope_i + b_vol(v) * lesion_volume_i + e_i(v)

FDR: Benjamini-Hochberg across in-mask voxels on the two-sided parametric p
     values of the hope t statistic, q < 0.05.
FWE: Freedman-Lane permutation of the hope regressor with lesion volume as
     nuisance, N_PERM permutations (the first is the unpermuted data), maximum
     |t| across the mask, two-sided, corrected p < 0.05.

Outputs (OUTPUT_DIR), each the t map with nonsurviving voxels set to 0:
  volumecontrolled_religious_hope_vhis_fdr05_t.nii.gz
  volumecontrolled_religious_hope_vhis_fwe05_t.nii.gz
  run_log_fdr_fwe.txt
"""
import os, time
import numpy as np, pandas as pd, nibabel as nib
from scipy import stats

OUTPUT_DIR = os.path.join(os.environ.get("VHIS_OUTPUT_DIR", "outputs"), "religious_hope_voxelwise_lnm")
T_MAP      = os.path.join(OUTPUT_DIR, "volumecontrolled_religious_hope_vhis_t_map.nii.gz")
N_PERM, SEED, ALPHA, BATCH = 5000, 20260920, 0.05, 50

logf = open(os.path.join(OUTPUT_DIR, "run_log_fdr_fwe.txt"), "w")
def log(m):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True); logf.write(line + "\n"); logf.flush()

sub = pd.read_csv(os.path.join(OUTPUT_DIR, "included_subjects.csv"))
n = len(sub)
ref = nib.load(T_MAP); shape = ref.shape[:3]; aff = ref.affine
t_saved = ref.get_fdata().reshape(-1)
D = np.stack([nib.load(f).get_fdata(dtype=np.float32).reshape(-1) for f in sub.network_file])
mask = np.all(np.isfinite(D), axis=0) & np.all(D != 0, axis=0)
mask &= D.var(axis=0) > 0
Y = D[:, mask].astype(np.float64); del D
nvox = Y.shape[1]
log(f"subjects {n}; in-mask voxels {nvox}")

x = sub.hope.values.astype(float); vol = sub.lesion_volume_mm3.values.astype(float)
Z = np.column_stack([np.ones(n), vol])
Qz, _ = np.linalg.qr(Z)
xr = x - Qz @ (Qz.T @ x); q3 = xr / np.linalg.norm(xr)
Q = np.column_stack([Qz, q3])
dof = n - 3
E = Y - Qz @ (Qz.T @ Y)
sst = (E * E).sum(axis=0)

def tstats(Qp):
    # Qp: (n, 3k) permuted orthonormal design columns, grouped per permutation as [z1, z2, x]
    A = Qp.T @ E
    A = A.reshape(-1, 3, nvox)
    sse = sst - (A * A).sum(axis=1)
    return A[:, 2, :] / np.sqrt(sse / dof)

t_obs = tstats(Q)[0]
assert np.allclose(t_obs, t_saved[mask], atol=1e-3), "observed t does not match saved t map"
log(f"observed t matches saved t map; t range {t_obs.min():.3f} to {t_obs.max():.3f}")

# FDR
p = 2 * stats.t.sf(np.abs(t_obs), dof)
order = np.argsort(p); ranked = p[order] * nvox / np.arange(1, nvox + 1)
qv = np.empty(nvox); qv[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1)
fdr = qv < ALPHA
log(f"FDR q<{ALPHA}: {int(fdr.sum())} voxels ({int((fdr & (t_obs>0)).sum())} positive, {int((fdr & (t_obs<0)).sum())} negative); min q {qv.min():.4f}"
    + (f"; |t| cutoff {np.abs(t_obs[fdr]).min():.3f}" if fdr.any() else ""))

# FWE
rng = np.random.default_rng(SEED)
maxt = np.empty(N_PERM); maxt[0] = np.abs(t_obs).max()
done = 1; t0 = time.time()
while done < N_PERM:
    k = min(BATCH, N_PERM - done)
    # permuting the rows of the data is the same as inverse-permuting the rows of the design
    Qp = np.concatenate([Q[rng.permutation(n)] for _ in range(k)], axis=1)
    maxt[done:done + k] = np.abs(tstats(Qp)).max(axis=1)
    done += k
    if done % 1000 < BATCH: log(f"  permutations {done}/{N_PERM} ({time.time()-t0:.0f}s)")
crit = np.quantile(maxt, 1 - ALPHA)
p_fwe = (maxt[None, :] >= np.abs(t_obs)[:, None] - 1e-12).mean(axis=1) if nvox * N_PERM < 5e7 else \
        1 - np.searchsorted(np.sort(maxt), np.abs(t_obs) - 1e-12, side="left") / N_PERM
fwe = p_fwe < ALPHA
log(f"FWE p<{ALPHA}: critical |t| {crit:.3f}; {int(fwe.sum())} voxels ({int((fwe & (t_obs>0)).sum())} positive, {int((fwe & (t_obs<0)).sum())} negative); min corrected p {p_fwe.min():.4f}")

for name, keep in [("fdr05", fdr), ("fwe05", fwe)]:
    full = np.zeros(int(np.prod(shape)), dtype=np.float32); full[mask] = np.where(keep, t_obs, 0)
    img = nib.Nifti1Image(full.reshape(shape), aff)
    out = os.path.join(OUTPUT_DIR, f"volumecontrolled_religious_hope_vhis_{name}_t.nii.gz")
    nib.save(img, out); log(f"saved {out}")
logf.close()
