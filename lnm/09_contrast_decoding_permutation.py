#!/usr/bin/env python3
"""
Permutation test of the meta-analytic decoding of the miracles versus magical
ideation contrast (Grafman VHIS cohort)

Question: is the spatial correspondence between the contrast map and the
decoded Neurosynth term maps larger than expected if belief in miracles and
magical ideation had the same relationship with lesion connectivity?

Model at every voxel v, across subjects i, with both scores z scored:
    N_i(v) = b0(v) + b_mir(v) * zmir_i + b_mag(v) * zmag_i + b_vol(v) * lesion_volume_i + e_i(v)

Contrast b_mir - b_mag. Reparameterized as
    N_i(v) = a(v) * (zmir_i - zmag_i) + c(v) * (zmir_i + zmag_i) + b0(v) + b_vol(v) * vol_i + e_i(v)
so that a = (b_mir - b_mag) / 2 and its t equals the contrast t. Freedman-Lane
permutation of the (zmir - zmag) regressor with intercept, lesion volume and
(zmir + zmag) as nuisance gives the null distribution of the contrast t map
under H0: b_mir = b_mag at every voxel. Permuting the rows of the orthonormal
design is equivalent to permuting the rows of the data (as in script 06).

Statistics, observed and in every permutation:
  r_term   Pearson r between the contrast t map and each term map, over voxels
           in the data mask that are nonzero in the term map
  r_mean   mean of r_term over the terms (omnibus)
  max |t|  for voxelwise FWE correction of the contrast t map
p values are two sided: the share of permutations (the observed one included)
with |statistic| at least the observed |statistic|.

Descriptive, not permuted: r between each term map and the t maps of zmir and
zmag from the same joint model.

Inputs:
  VHIS_DATA_DIR/miracles_magical_network_matched.xlsx (sheet matched):
      subject_id, miracles_score, magical_ideation_total,
      network_file_full_path, lesion_file_full_path
  NEUROSYNTH_TERM_DIR: Neurosynth association test z maps named
      terms_abstract_tfidf__<term>.z_desc-association.nii.gz
      (resampled to the data grid, which also corrects any axis flip)
Optional: PATH_PREFIX_MAP=old_prefix=new_prefix to remap image paths.

Outputs (OUTPUT_DIR):
  contrast_decoding_permutation.xlsx
  contrast_tstat_zscored.nii.gz, contrast_tstat_zscored_fwe05.nii.gz
  run_log.txt
"""
import os, time
import numpy as np, pandas as pd, nibabel as nib
from nibabel.processing import resample_from_to

DATA_DIR    = os.environ["VHIS_DATA_DIR"]
TERM_DIR    = os.environ["NEUROSYNTH_TERM_DIR"]
OUTPUT_ROOT = os.environ.get("VHIS_OUTPUT_DIR", "outputs")
PREFIX_MAP  = os.environ.get("PATH_PREFIX_MAP", "")
SPREADSHEET = os.path.join(DATA_DIR, "miracles_magical_network_matched.xlsx")
OUTPUT_DIR  = os.path.join(OUTPUT_ROOT, "contrast_decoding_permutation")
TERMS = ["medial prefrontal", "medial", "prefrontal", "theory mind", "prefrontal cortex",
         "default", "autobiographical", "mind", "mind tom", "default mode"]   # ten top decoded terms in the paper
N_PERM, SEED, ALPHA, BATCH = int(os.environ.get("N_PERM", 5000)), 20261019, 0.05, 20

os.makedirs(OUTPUT_DIR, exist_ok=True)
logf = open(os.path.join(OUTPUT_DIR, "run_log.txt"), "w")
def log(m):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True); logf.write(line + "\n"); logf.flush()

def remap(p):
    if PREFIX_MAP:
        old, new = PREFIX_MAP.split("=", 1)
        if p.startswith(old): return new + p[len(old):]
    return p

log(f"script: {os.path.abspath(__file__)}")
log(f"scores: {SPREADSHEET}"); log(f"terms: {TERM_DIR}"); log(f"permutations: {N_PERM}, seed {SEED}")

sub = pd.read_excel(SPREADSHEET, sheet_name="matched")
sub["network_file"] = sub.network_file_full_path.map(remap)
sub["lesion_file"] = sub.lesion_file_full_path.map(remap)
n = len(sub)
ref = nib.load(sub.network_file.iloc[0]); shape = ref.shape[:3]; aff = ref.affine
nv = int(np.prod(shape))
D = np.zeros((n, nv), dtype=np.float32); vol = np.zeros(n)
for i, r in enumerate(sub.itertuples()):
    im = nib.load(r.network_file)
    assert im.shape[:3] == shape and np.allclose(im.affine, aff), r.network_file
    D[i] = im.get_fdata(dtype=np.float32).reshape(-1)
    L = nib.load(r.lesion_file); assert L.shape[:3] == shape
    vol[i] = (L.get_fdata() > 0).sum() * float(abs(np.linalg.det(L.affine[:3, :3])))
mir = sub.miracles_score.values.astype(float); mag = sub.magical_ideation_total.values.astype(float)
log(f"subjects {n}; miracles mean {mir.mean():.3f} sd {mir.std(ddof=1):.3f}; magical ideation mean {mag.mean():.3f} sd {mag.std(ddof=1):.3f}; r = {np.corrcoef(mir, mag)[0,1]:.3f}")

mask = np.all(np.isfinite(D), axis=0) & np.all(D != 0, axis=0)
mask &= D.var(axis=0) > 0
Y = D[:, mask].astype(np.float64); del D
nvox = Y.shape[1]
log(f"in-mask voxels (nonzero in every subject): {nvox}")

refimg = nib.Nifti1Image(np.zeros(shape, np.float32), aff)
Tm, tok = [], []
for t in TERMS:
    f = os.path.join(TERM_DIR, f"terms_abstract_tfidf__{t}.z_desc-association.nii.gz")
    tm = resample_from_to(nib.load(f), refimg, order=0).get_fdata().reshape(-1)[mask]
    Tm.append(tm); tok.append(np.isfinite(tm) & (tm != 0))
Tm = np.array(Tm); tok = np.array(tok)
log(f"term maps resampled to data grid; voxels used per term {tok.sum(axis=1).min()} to {tok.sum(axis=1).max()}")

def spatial_r(tmaps):
    # tmaps: (k, nvox) -> (k, n_terms) Pearson r over each term's valid voxels
    out = np.empty((tmaps.shape[0], len(TERMS)))
    for j in range(len(TERMS)):
        a = tmaps[:, tok[j]]; b = Tm[j, tok[j]]
        a = a - a.mean(axis=1, keepdims=True); b = b - b.mean()
        out[:, j] = (a @ b) / (np.sqrt((a * a).sum(axis=1)) * np.sqrt((b * b).sum()))
    return out

z = lambda x: (x - x.mean()) / x.std(ddof=1)
zmir, zmag = z(mir), z(mag)
dif, tot = zmir - zmag, zmir + zmag
Z = np.column_stack([np.ones(n), vol, tot])
Qz, _ = np.linalg.qr(Z)
xr = dif - Qz @ (Qz.T @ dif); q = xr / np.linalg.norm(xr)
Q = np.column_stack([Qz, q]); p_cols = Q.shape[1]
dof = n - p_cols
E = Y - Qz @ (Qz.T @ Y)
sst = (E * E).sum(axis=0)

def tstats(Qp):
    # Qp: (n, p_cols * k), permuted orthonormal design columns grouped per permutation
    A = (Qp.T @ E).reshape(-1, p_cols, nvox)
    sse = sst - (A * A).sum(axis=1)
    return A[:, -1, :] / np.sqrt(sse / dof)

t_obs = tstats(Q)[0]

# check against an explicit fit of the unreparameterized model
M = np.column_stack([np.ones(n), zmir, zmag, vol])
B = np.linalg.lstsq(M, Y, rcond=None)[0]; R = Y - M @ B
s2 = (R * R).sum(axis=0) / (n - 4); Minv = np.linalg.inv(M.T @ M)
c = np.array([0, 1, -1, 0.0])
t_check = (c @ B) / np.sqrt(s2 * (c @ Minv @ c))
assert np.allclose(t_obs, t_check, atol=1e-6), "reparameterized t does not match the explicit contrast"
t_mir = B[1] / np.sqrt(s2 * Minv[1, 1]); t_mag = B[2] / np.sqrt(s2 * Minv[2, 2])
log(f"contrast t matches explicit model; t range {t_obs.min():.3f} to {t_obs.max():.3f}")

r_obs = spatial_r(t_obs[None])[0]
r_mir, r_mag = spatial_r(t_mir[None])[0], spatial_r(t_mag[None])[0]
log("observed spatial r (contrast | miracles alone | magical ideation alone, joint model):")
for j, t in enumerate(TERMS):
    log(f"  {t:20s} {r_obs[j]:+.3f} | {r_mir[j]:+.3f} | {r_mag[j]:+.3f}")
log(f"observed mean r over terms {r_obs.mean():+.3f}; spatial r(contrast, miracles t) = {np.corrcoef(t_obs, t_mir)[0,1]:.3f}, r(contrast, magical t) = {np.corrcoef(t_obs, t_mag)[0,1]:.3f}")

rng = np.random.default_rng(SEED)
null_r = np.empty((N_PERM, len(TERMS))); maxt = np.empty(N_PERM)
null_r[0] = r_obs; maxt[0] = np.abs(t_obs).max()
done = 1; t0 = time.time()
while done < N_PERM:
    k = min(BATCH, N_PERM - done)
    Qp = np.concatenate([Q[rng.permutation(n)] for _ in range(k)], axis=1)
    tp = tstats(Qp)
    null_r[done:done + k] = spatial_r(tp); maxt[done:done + k] = np.abs(tp).max(axis=1)
    done += k
    if done % 500 < BATCH: log(f"  permutations {done}/{N_PERM} ({time.time()-t0:.0f}s)")

p_term = (np.abs(null_r) >= np.abs(r_obs)[None] - 1e-12).mean(axis=0)
null_mean = null_r.mean(axis=1)
p_mean = (np.abs(null_mean) >= abs(r_obs.mean()) - 1e-12).mean()
crit = np.quantile(maxt, 1 - ALPHA)
p_fwe = 1 - np.searchsorted(np.sort(maxt), np.abs(t_obs) - 1e-12, side="left") / N_PERM
fwe = p_fwe < ALPHA
log("permutation p (two sided) per term:")
for j, t in enumerate(TERMS):
    lo, hi = np.quantile(null_r[:, j], [0.025, 0.975])
    log(f"  {t:20s} r {r_obs[j]:+.3f}  null 95% [{lo:+.3f}, {hi:+.3f}]  p {p_term[j]:.4f}")
log(f"omnibus mean r {r_obs.mean():+.3f}  null 95% [{np.quantile(null_mean,0.025):+.3f}, {np.quantile(null_mean,0.975):+.3f}]  p {p_mean:.4f}")
log(f"voxelwise FWE p<{ALPHA}: critical |t| {crit:.3f}; {int(fwe.sum())} voxels ({int((fwe & (t_obs>0)).sum())} positive, {int((fwe & (t_obs<0)).sum())} negative); min corrected p {p_fwe.min():.4f}")

for name, vals in [("contrast_tstat_zscored", t_obs), ("contrast_tstat_zscored_fwe05", np.where(fwe, t_obs, 0))]:
    full = np.zeros(nv, dtype=np.float32); full[mask] = vals
    nib.save(nib.Nifti1Image(full.reshape(shape), aff), os.path.join(OUTPUT_DIR, f"{name}.nii.gz"))
summary = pd.DataFrame({"term": TERMS, "r_contrast": r_obs,
                        "null_2.5": np.quantile(null_r, 0.025, axis=0), "null_97.5": np.quantile(null_r, 0.975, axis=0),
                        "p_two_sided": p_term, "r_miracles_joint": r_mir, "r_magical_joint": r_mag})
summary.loc[len(summary)] = ["MEAN OF TERMS", r_obs.mean(), np.quantile(null_mean, 0.025), np.quantile(null_mean, 0.975),
                             p_mean, r_mir.mean(), r_mag.mean()]
with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "contrast_decoding_permutation.xlsx")) as w:
    summary.to_excel(w, sheet_name="summary", index=False)
    pd.DataFrame(null_r, columns=TERMS).to_excel(w, sheet_name="null_r", index=False)
log(f"saved outputs to {OUTPUT_DIR}"); logf.close()
