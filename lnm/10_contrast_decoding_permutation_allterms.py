#!/usr/bin/env python3
"""
Selection-corrected permutation test of the meta-analytic decoding of the
miracles versus magical ideation contrast (Grafman VHIS cohort)

Script 09 tests the ten terms that ranked highest for the observed contrast
map. Because those terms were chosen on the same map, that test is partly
circular. Here the decoding is repeated in full inside every permutation:
each permuted contrast map is correlated with every term map, and the terms
are reranked, so the null distribution includes the selection step.

Model, contrast and Freedman-Lane permutation scheme as in script 09:
    N_i(v) = b0(v) + b_mir(v) * zmir_i + b_mag(v) * zmag_i + b_vol(v) * lesion_volume_i + e_i(v)
contrast b_mir - b_mag, H0: b_mir = b_mag at every voxel.

Statistics, observed and in every permutation, over all term maps:
  top10    mean |r| of the ten terms with the largest |r|
  max      largest |r| over terms
For each term, the FWE-corrected p is the share of permutations (observed
included) whose max |r| is at least the term's observed |r|. Spatial r is
Pearson r over the data mask (voxels nonzero in every subject's network map)
restricted to voxels that are finite and nonzero in every term map.

Inputs:
  VHIS_DATA_DIR/miracles_magical_network_matched.xlsx (sheet matched), as in script 09
  NEUROSYNTH_TERM_DIR: all Neurosynth association test z maps,
      terms_abstract_tfidf__<term>.z_desc-association.nii.gz
Optional: PATH_PREFIX_MAP=old_prefix=new_prefix; N_PERM (default 5000).

Outputs (OUTPUT_DIR):
  contrast_decoding_permutation_allterms.xlsx
  run_log.txt
"""
import os, glob, time
import numpy as np, pandas as pd, nibabel as nib
from nibabel.processing import resample_from_to

DATA_DIR    = os.environ["VHIS_DATA_DIR"]
TERM_DIR    = os.environ["NEUROSYNTH_TERM_DIR"]
OUTPUT_ROOT = os.environ.get("VHIS_OUTPUT_DIR", "outputs")
PREFIX_MAP  = os.environ.get("PATH_PREFIX_MAP", "")
SPREADSHEET = os.path.join(DATA_DIR, "miracles_magical_network_matched.xlsx")
OUTPUT_DIR  = os.path.join(OUTPUT_ROOT, "contrast_decoding_permutation_allterms")
PAPER_TERMS = ["medial prefrontal", "medial", "prefrontal", "theory mind", "prefrontal cortex",
               "default", "autobiographical", "mind", "mind tom", "default mode"]
N_PERM, SEED, ALPHA, BATCH, TOPK = int(os.environ.get("N_PERM", 5000)), 20261019, 0.05, 20, 10

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
mask = np.all(np.isfinite(D), axis=0) & np.all(D != 0, axis=0)
mask &= D.var(axis=0) > 0
log(f"subjects {n}; in-mask voxels {int(mask.sum())}")

# term maps, resampled to the data grid (corrects axis flips)
refimg = nib.Nifti1Image(np.zeros(shape, np.float32), aff)
files = sorted(glob.glob(os.path.join(TERM_DIR, "terms_abstract_tfidf__*.z_desc-association.nii.gz")))
names = [os.path.basename(f).split("__", 1)[1].split(".z_desc")[0] for f in files]
# resampling every map is slow, so check once whether maps on the first map's
# grid only need their x axis reversed (or nothing) to match the data grid
first = nib.load(files[0]); raw0 = np.asarray(first.dataobj, dtype=np.float32)
res0 = resample_from_to(first, refimg, order=0).get_fdata(dtype=np.float32)
fast = None
if raw0.shape == shape:
    if np.array_equal(raw0[::-1], res0): fast = "flip"
    elif np.array_equal(raw0, res0): fast = "same"
Tall = np.empty((len(files), int(mask.sum())), dtype=np.float32); tvalid = np.ones(int(mask.sum()), bool)
t0 = time.time()
for j, f in enumerate(files):
    im = nib.load(f)
    if fast and im.shape[:3] == shape and np.allclose(im.affine, first.affine):
        arr = np.asarray(im.dataobj, dtype=np.float32)
        if fast == "flip": arr = arr[::-1]
    else:
        arr = resample_from_to(im, refimg, order=0).get_fdata(dtype=np.float32)
    Tall[j] = arr.reshape(-1)[mask]
    tvalid &= np.isfinite(Tall[j]) & (Tall[j] != 0)
    if (j + 1) % 500 == 0: log(f"  loaded {j+1}/{len(files)} term maps ({time.time()-t0:.0f}s)")
log(f"{len(files)} term maps loaded (fast path: {fast}); voxels valid in every term map: {int(tvalid.sum())}")
missing = [t for t in PAPER_TERMS if t not in names]
assert not missing, f"paper terms not found: {missing}"

Y = D[:, mask][:, tvalid].astype(np.float64); del D
nvox = Y.shape[1]
Tz = Tall[:, tvalid].astype(np.float64); del Tall
Tz -= Tz.mean(axis=1, keepdims=True); Tz /= np.linalg.norm(Tz, axis=1, keepdims=True)
Tz = Tz.astype(np.float32)

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
    A = (Qp.T @ E).reshape(-1, p_cols, nvox)
    sse = sst - (A * A).sum(axis=1)
    return A[:, -1, :] / np.sqrt(sse / dof)

def decode(tmaps):
    # (k, nvox) -> (k, n_terms) Pearson r
    tc = tmaps - tmaps.mean(axis=1, keepdims=True); tc /= np.linalg.norm(tc, axis=1, keepdims=True)
    return tc.astype(np.float32) @ Tz.T

t_obs = tstats(Q)[0]
r_obs = decode(t_obs[None])[0].astype(float)
a_obs = np.abs(r_obs)
order = np.argsort(-a_obs)
top10_obs = a_obs[order[:TOPK]].mean(); max_obs = a_obs.max()
log("observed top terms by |r| (all terms):")
for j in order[:20]: log(f"  {names[j]:30s} r {r_obs[j]:+.3f}")
log("paper's ten terms: " + ", ".join(f"{t} {r_obs[names.index(t)]:+.3f} (rank {int(np.where(order == names.index(t))[0][0]) + 1})" for t in PAPER_TERMS))

rng = np.random.default_rng(SEED)
null_top10 = np.empty(N_PERM); null_max = np.empty(N_PERM)
null_top10[0], null_max[0] = top10_obs, max_obs
done = 1; t0 = time.time()
while done < N_PERM:
    k = min(BATCH, N_PERM - done)
    Qp = np.concatenate([Q[rng.permutation(n)] for _ in range(k)], axis=1)
    a = np.abs(decode(tstats(Qp)))
    null_max[done:done + k] = a.max(axis=1)
    null_top10[done:done + k] = -np.sort(-a, axis=1)[:, :TOPK].mean(axis=1)
    done += k
    if done % 500 < BATCH: log(f"  permutations {done}/{N_PERM} ({time.time()-t0:.0f}s)")

p_top10 = (null_top10 >= top10_obs - 1e-9).mean()
p_max = (null_max >= max_obs - 1e-9).mean()
sm = np.sort(null_max)
p_fwe = 1 - np.searchsorted(sm, a_obs - 1e-9, side="left") / N_PERM
log(f"top-{TOPK} mean |r|: observed {top10_obs:.3f}; null median {np.median(null_top10):.3f}, 95th percentile {np.quantile(null_top10, 0.95):.3f}; p {p_top10:.4f}")
log(f"max |r|: observed {max_obs:.3f} ({names[order[0]]}); null median {np.median(null_max):.3f}, 95th percentile {np.quantile(null_max, 0.95):.3f}; p {p_max:.4f}")
log(f"terms with FWE-corrected p < {ALPHA}: {int((p_fwe < ALPHA).sum())}")
for t in PAPER_TERMS:
    j = names.index(t); log(f"  {t:20s} r {r_obs[j]:+.3f}  FWE-corrected p {p_fwe[j]:.4f}")

res = pd.DataFrame({"term": names, "r": r_obs, "abs_r": a_obs, "rank": np.argsort(order) + 1, "p_fwe": p_fwe,
                    "in_paper_top10": [t in PAPER_TERMS for t in names]}).sort_values("rank")
summ = pd.DataFrame([
    {"statistic": f"top-{TOPK} mean |r|", "observed": top10_obs, "null_median": np.median(null_top10),
     "null_95th": np.quantile(null_top10, 0.95), "p": p_top10},
    {"statistic": "max |r|", "observed": max_obs, "null_median": np.median(null_max),
     "null_95th": np.quantile(null_max, 0.95), "p": p_max}])
with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "contrast_decoding_permutation_allterms.xlsx")) as w:
    summ.to_excel(w, sheet_name="summary", index=False)
    res.to_excel(w, sheet_name="all_terms", index=False)
    pd.DataFrame({"null_top10_mean_abs_r": null_top10, "null_max_abs_r": null_max}).to_excel(w, sheet_name="null", index=False)
log(f"saved outputs to {OUTPUT_DIR}"); logf.close()
