#!/usr/bin/env python3
"""
Voxelwise contrast: belief in miracles vs. magical ideation
===========================================================

Computes an UNTHRESHOLDED map of the contrast (beta_mir - beta_mag) across
lesion network connectivity maps, for the VHIS cohort of 106 subjects with
complete data (network file + miracle score + magical ideation total).

MODEL (fit independently at every voxel v, across subjects):
    N_i(v) = b0(v) + b_mir(v) * mir_i + b_mag(v) * mag_i + e_i(v)

CONTRAST:
    beta_mir(v) - beta_mag(v)

SIGN CONVENTION (read this before interpreting any output):
    Each subject's input map N_i(v) is the t-statistic of lesion-to-voxel
    functional connectivity from the normative connectome. Under the LNM
    convention, a positive value of N_i(v) at a voxel means the lesion is
    positively connected to that voxel, which means lesion-driven disruption
    propagates to that voxel; since lesions decrease activity in functionally
    connected regions, positive connectivity -> disruption.

    Therefore:
      * POSITIVE contrast value at voxel v  -> b_mir > b_mag
            -> disruption of v is MORE coupled to miracle belief than to
               magical ideation
            -> v's intact activity is MORE suppressive of miracle belief
               than of magical ideation
            -> equivalently, v is MORE engaged during magical ideation than
               during miracle belief

      * NEGATIVE contrast value at voxel v  -> b_mag > b_mir
            -> disruption of v is MORE coupled to magical ideation than to
               miracle belief
            -> v's intact activity is MORE suppressive of magical ideation
               than of miracle belief
            -> equivalently, v is MORE engaged during miracle belief than
               during magical ideation

    Hypothesis being tested: if miracle belief specifically recruits theory
    of mind circuitry (mPFC, TPJ, PCC, etc.), those regions should fall on
    the NEGATIVE side of this contrast.

No thresholding, no permutation inference, no covariates. This is a
descriptive map, intended to be submitted to Neurosynth Decode as-is.

INPUTS:
    - An xlsx with columns: subject_id, miracles_score, magical_ideation_total,
      network_file_full_path. (Produced by the matching step.)
    - The NIfTI lesion-network t-maps, one per subject, at the paths in the
      xlsx. All in the same MNI space, same dimensions, same affine.

OUTPUTS (written to --outdir):
    contrast_beta_mir_minus_mag.nii.gz     unthresholded contrast map
    contrast_tstat_mir_minus_mag.nii.gz    t-stat for the same contrast
    beta_miracles.nii.gz                   beta for miracles predictor
    beta_magical.nii.gz                    beta for magical ideation predictor
    r_squared.nii.gz                       voxelwise R^2 (QC)
    run_log.txt                            full audit trail of the run

USAGE:
    python miracles_vs_magical_contrast.py \
        --xlsx /path/to/miracles_magical_network_matched.xlsx \
        --outdir /path/to/output_directory

    # If paths in the xlsx no longer match where the NIfTIs actually live,
    # use --path-prefix-map old_prefix=new_prefix to remap on the fly:
    python miracles_vs_magical_contrast.py \
        --xlsx ... --outdir ... \
        --path-prefix-map /old/location=/new/location

DEPENDENCIES:
    numpy, pandas, nibabel, openpyxl, scipy
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats


# -----------------------------------------------------------------------------
# Configuration: anchor ROIs for the sign-verification block
# -----------------------------------------------------------------------------
# After the map is computed, we sample the contrast value at a handful of
# canonical coordinates to verify the sign makes sense before the map is
# uploaded anywhere. Coordinates are in MNI (mm), will be mapped to voxel
# space via each subject's affine. Radius is in mm.
ANCHOR_ROIS = [
    # (label, x_mm, y_mm, z_mm, radius_mm, expected_sign_under_hypothesis)
    ("right TPJ",              54, -54,  22, 8, "negative"),
    ("left TPJ",              -54, -58,  22, 8, "negative"),
    ("mPFC",                    0,  54,  10, 8, "negative"),
    ("PCC",                     0, -54,  30, 8, "negative"),
    ("right cerebellar VI",    26, -66, -26, 8, "negative"),
    # A comparison region that should NOT show a strong effect either way:
    ("primary motor cortex",  -38, -22,  54, 8, "null (control)"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xlsx", required=True,
                   help="Path to miracles_magical_network_matched.xlsx")
    p.add_argument("--outdir", required=True,
                   help="Directory to write output NIfTI maps and log")
    p.add_argument("--sheet", default="matched",
                   help="Sheet name in the xlsx (default: matched)")
    p.add_argument("--path-prefix-map", default=None,
                   help="Remap NIfTI paths: old_prefix=new_prefix")
    p.add_argument("--mask", default=None,
                   help="Optional brain mask NIfTI; voxels outside are set to 0")
    return p.parse_args()


def remap_path(path, prefix_map):
    if not prefix_map:
        return path
    old, new = prefix_map.split("=", 1)
    if path.startswith(old):
        return new + path[len(old):]
    return path


def log(msg, fh=None):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log_path = outdir / "run_log.txt"
    fh = open(log_path, "w")

    log(f"Script: {Path(__file__).resolve()}", fh)
    log(f"Working directory: {os.getcwd()}", fh)
    log(f"xlsx: {args.xlsx}", fh)
    log(f"outdir: {outdir.resolve()}", fh)
    log(f"sheet: {args.sheet}", fh)
    log(f"path-prefix-map: {args.path_prefix_map}", fh)
    log(f"mask: {args.mask}", fh)

    # -------------------------------------------------------------------------
    # 1. Load cohort
    # -------------------------------------------------------------------------
    df = pd.read_excel(args.xlsx, sheet_name=args.sheet)
    required_cols = {"subject_id", "miracles_score", "magical_ideation_total",
                     "network_file_full_path"}
    missing = required_cols - set(df.columns)
    if missing:
        log(f"ERROR: xlsx missing columns: {missing}", fh)
        sys.exit(1)

    # Remap paths if needed
    df["network_file_resolved"] = df["network_file_full_path"].apply(
        lambda p: remap_path(p, args.path_prefix_map))

    # Check files exist
    missing_files = df[~df["network_file_resolved"].map(os.path.exists)]
    if len(missing_files):
        log(f"ERROR: {len(missing_files)} network files not found. Examples:", fh)
        for _, r in missing_files.head(5).iterrows():
            log(f"  subject {r['subject_id']}: {r['network_file_resolved']}", fh)
        log("Use --path-prefix-map to remap paths if they have moved.", fh)
        sys.exit(1)

    n = len(df)
    log(f"Loaded {n} subjects from xlsx.", fh)
    log(f"Miracles score: mean={df['miracles_score'].mean():.3f}, "
        f"sd={df['miracles_score'].std():.3f}, "
        f"range=[{df['miracles_score'].min()}, {df['miracles_score'].max()}]", fh)
    log(f"Magical ideation total: mean={df['magical_ideation_total'].mean():.3f}, "
        f"sd={df['magical_ideation_total'].std():.3f}, "
        f"range=[{df['magical_ideation_total'].min()}, "
        f"{df['magical_ideation_total'].max()}]", fh)
    log(f"Pearson r(miracles, magical) = "
        f"{df['miracles_score'].corr(df['magical_ideation_total']):.4f}", fh)

    # -------------------------------------------------------------------------
    # 2. Load first NIfTI to establish dimensions + affine
    # -------------------------------------------------------------------------
    first_path = df["network_file_resolved"].iloc[0]
    log(f"Loading reference NIfTI: {first_path}", fh)
    ref_img = nib.load(first_path)
    ref_shape = ref_img.shape
    ref_affine = ref_img.affine
    log(f"Reference shape: {ref_shape}", fh)
    log(f"Reference affine:\n{ref_affine}", fh)
    if len(ref_shape) != 3:
        log(f"ERROR: expected 3D image, got shape {ref_shape}", fh)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 3. Stack all subject maps into a 4D array (X, Y, Z, N)
    # -------------------------------------------------------------------------
    log(f"Allocating 4D array: {ref_shape + (n,)} float32 "
        f"(~{np.prod(ref_shape)*n*4/1e9:.2f} GB)", fh)
    data4d = np.empty(ref_shape + (n,), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        path = row["network_file_resolved"]
        img = nib.load(path)
        if img.shape != ref_shape:
            log(f"ERROR: subject {row['subject_id']} shape {img.shape} "
                f"!= reference {ref_shape}", fh)
            sys.exit(1)
        if not np.allclose(img.affine, ref_affine, atol=1e-4):
            log(f"WARNING: subject {row['subject_id']} affine differs from "
                f"reference. Proceeding, but verify alignment.", fh)
        data4d[..., i] = np.asarray(img.dataobj, dtype=np.float32)
        if (i + 1) % 20 == 0 or (i + 1) == n:
            log(f"  loaded {i+1}/{n}", fh)

    # Replace any NaN or Inf with 0 (these are outside-brain voxels)
    nonfinite = ~np.isfinite(data4d)
    if nonfinite.any():
        log(f"Found {nonfinite.sum()} non-finite voxel-values; setting to 0", fh)
        data4d[nonfinite] = 0.0

    # -------------------------------------------------------------------------
    # 4. Build design matrix and fit OLS at every voxel (vectorized)
    # -------------------------------------------------------------------------
    # Design: [intercept, miracles, magical]
    mir = df["miracles_score"].to_numpy(dtype=np.float64)
    mag = df["magical_ideation_total"].to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones(n), mir, mag])  # shape (n, 3)
    log(f"Design matrix X shape: {X.shape}", fh)
    log(f"Design matrix rank: {np.linalg.matrix_rank(X)} (should be 3)", fh)

    # Reshape 4D -> (n_voxels, n_subjects)
    nx, ny, nz = ref_shape
    n_vox = nx * ny * nz
    Y = data4d.reshape(n_vox, n).astype(np.float64)  # (n_vox, n)
    # We need Y.T in the regression: (n, n_vox)
    # OLS: beta = (X' X)^-1 X' y   for each voxel independently
    # Vectorize across voxels:
    XtX_inv = np.linalg.inv(X.T @ X)            # (3, 3)
    XtY = X.T @ Y.T                              # (3, n_vox)
    betas = XtX_inv @ XtY                        # (3, n_vox)
    b0 = betas[0, :]
    b_mir = betas[1, :]
    b_mag = betas[2, :]

    # Residuals and standard error for the contrast
    Y_hat = (X @ betas).T                        # (n_vox, n)
    resid = Y - Y_hat                            # (n_vox, n)
    df_resid = n - X.shape[1]                    # degrees of freedom
    # Mean-squared error per voxel
    mse = np.sum(resid ** 2, axis=1) / df_resid  # (n_vox,)

    # Contrast c = [0, +1, -1], so Var(c'b) = c' (X'X)^-1 c * sigma^2
    c = np.array([0.0, 1.0, -1.0])
    var_c = float(c @ XtX_inv @ c)               # scalar (design-only term)
    contrast_val = b_mir - b_mag                 # (n_vox,)
    se_contrast = np.sqrt(var_c * mse)           # (n_vox,)
    # Avoid division by zero outside brain
    with np.errstate(divide="ignore", invalid="ignore"):
        t_contrast = np.where(se_contrast > 0,
                              contrast_val / se_contrast,
                              0.0)

    # Voxelwise R^2 (QC)
    ss_total = np.sum((Y - Y.mean(axis=1, keepdims=True)) ** 2, axis=1)
    ss_resid = np.sum(resid ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_total > 0, 1.0 - ss_resid / ss_total, 0.0)

    # Apply optional mask
    if args.mask:
        log(f"Applying mask: {args.mask}", fh)
        mask_img = nib.load(args.mask)
        if mask_img.shape != ref_shape:
            log(f"ERROR: mask shape {mask_img.shape} != data shape {ref_shape}",
                fh)
            sys.exit(1)
        mask = (np.asarray(mask_img.dataobj) > 0).reshape(n_vox)
        for arr in (contrast_val, t_contrast, b_mir, b_mag, r2):
            arr[~mask] = 0.0

    # -------------------------------------------------------------------------
    # 5. Save NIfTI outputs
    # -------------------------------------------------------------------------
    def save(arr_flat, name):
        arr3d = arr_flat.reshape(ref_shape).astype(np.float32)
        out_img = nib.Nifti1Image(arr3d, ref_affine, ref_img.header)
        out_img.set_data_dtype(np.float32)
        out_path = outdir / name
        nib.save(out_img, out_path)
        log(f"  wrote {out_path}  (min={arr3d.min():.4f}, "
            f"max={arr3d.max():.4f}, mean={arr3d.mean():.4f})", fh)

    log("Saving outputs:", fh)
    save(contrast_val, "contrast_beta_mir_minus_mag.nii.gz")
    save(t_contrast,   "contrast_tstat_mir_minus_mag.nii.gz")
    save(b_mir,        "beta_miracles.nii.gz")
    save(b_mag,        "beta_magical.nii.gz")
    save(r2,           "r_squared.nii.gz")

    # -------------------------------------------------------------------------
    # 6. Sign verification at anchor ROIs
    # -------------------------------------------------------------------------
    log("", fh)
    log("=" * 70, fh)
    log("SIGN VERIFICATION: contrast value averaged within canonical ROIs", fh)
    log("=" * 70, fh)
    log("Under the hypothesis that miracle belief specifically recruits theory", fh)
    log("of mind circuitry, ToM regions should show NEGATIVE contrast values", fh)
    log("(disruption of these regions is more coupled to magical ideation,", fh)
    log(" meaning their intact engagement is more coupled to miracle belief).", fh)
    log("", fh)

    contrast_3d = contrast_val.reshape(ref_shape)
    tstat_3d = t_contrast.reshape(ref_shape)
    inv_affine = np.linalg.inv(ref_affine)

    def mni_to_voxel(x, y, z):
        """Map MNI mm -> voxel index using the image affine."""
        mni = np.array([x, y, z, 1.0])
        vox = inv_affine @ mni
        return int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))

    def sphere_mean(arr3d, cx, cy, cz, radius_mm):
        """Mean of arr3d within a sphere of radius_mm centered at MNI (cx,cy,cz)."""
        vx, vy, vz = mni_to_voxel(cx, cy, cz)
        # voxel-space radius in each direction (assumes isotropic or near-iso)
        # use the smallest voxel dimension to be conservative
        voxel_sizes = np.abs(np.diag(ref_affine)[:3])
        r_vox = int(np.ceil(radius_mm / voxel_sizes.min()))
        rr = range(-r_vox, r_vox + 1)
        vals = []
        for dx in rr:
            for dy in rr:
                for dz in rr:
                    # check spherical constraint in mm
                    dist_mm = np.sqrt((dx*voxel_sizes[0])**2
                                       + (dy*voxel_sizes[1])**2
                                       + (dz*voxel_sizes[2])**2)
                    if dist_mm > radius_mm:
                        continue
                    ix, iy, iz = vx + dx, vy + dy, vz + dz
                    if (0 <= ix < arr3d.shape[0]
                            and 0 <= iy < arr3d.shape[1]
                            and 0 <= iz < arr3d.shape[2]):
                        vals.append(arr3d[ix, iy, iz])
        if not vals:
            return float("nan"), 0
        return float(np.mean(vals)), len(vals)

    log(f"{'ROI':28s} {'MNI':>16s} {'contrast':>12s} {'t-stat':>10s} "
        f"{'n_vox':>8s}  expected", fh)
    log("-" * 95, fh)
    for label, cx, cy, cz, r_mm, expected in ANCHOR_ROIS:
        c_mean, nv = sphere_mean(contrast_3d, cx, cy, cz, r_mm)
        t_mean, _ = sphere_mean(tstat_3d, cx, cy, cz, r_mm)
        mni_str = f"({cx:+d},{cy:+d},{cz:+d})"
        log(f"{label:28s} {mni_str:>16s} {c_mean:+12.4f} {t_mean:+10.3f} "
            f"{nv:>8d}  {expected}", fh)

    log("", fh)
    log("INTERPRETATION GUIDE:", fh)
    log("  * Positive contrast at ToM voxels would CONTRADICT the hypothesis", fh)
    log("    (would mean miracle belief is LESS ToM-coupled than magical ideation)", fh)
    log("  * Negative contrast at ToM voxels would SUPPORT the hypothesis", fh)
    log("    (would mean miracle belief is MORE ToM-coupled than magical ideation)", fh)
    log("  * Values near zero at primary motor cortex are a null-control check", fh)
    log("", fh)
    log("Next step: upload contrast_beta_mir_minus_mag.nii.gz to Neurosynth", fh)
    log("  Decoder (https://neurosynth.org/decode/). Note that the decoder", fh)
    log("  correlates spatially and is sign-BLIND with respect to which", fh)
    log("  regions are positive vs negative in your map. The decoder tells", fh)
    log("  you WHICH functional terms your map's spatial pattern matches;", fh)
    log("  the SIGN of the pattern-matching (miracle-side vs magical-side)", fh)
    log("  must be read off the sign-verification block above.", fh)
    log("", fh)
    log("Done.", fh)
    fh.close()
    print(f"\nLog written to: {log_path}")


if __name__ == "__main__":
    main()
