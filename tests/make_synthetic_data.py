#!/usr/bin/env python3
"""
Synthetic stand-in for the VHIS inputs, used only to check that the pipeline runs

Writes random 12 x 12 x 12 lesion masks and lesion network maps for 40 fake
subjects, plus spreadsheets with the same file names, sheet names and columns
as the real (non-public) behavioral files. Two IDs have scores but no images,
to exercise the exclusion branch. Nothing here resembles real participant data.

Usage: python make_synthetic_data.py <output_dir>
Creates <output_dir>/images and <output_dir>/data.
"""
import os, sys
import numpy as np, pandas as pd, nibabel as nib

root = sys.argv[1]
img_dir, dat_dir = os.path.join(root, "images"), os.path.join(root, "data")
os.makedirs(img_dir, exist_ok=True); os.makedirs(dat_dir, exist_ok=True)
rng = np.random.default_rng(0)
shape, aff = (12, 12, 12), np.diag([-2.0, 2.0, 2.0, 1.0])
ids = list(range(1001, 1021)) + list(range(5001, 5021))
for i in ids:
    les = np.zeros(shape, np.uint8)
    c = rng.integers(2, 10, 3); les[c[0]-1:c[0]+2, c[1]-1:c[1]+2, c[2]-rng.integers(0, 2):c[2]+2] = 1
    nib.save(nib.Nifti1Image(les, aff), os.path.join(img_dir, f"{i:04d}.nii"))
    net = rng.normal(0, 3, shape).astype(np.float32) + 0.1
    nib.save(nib.Nifti1Image(net, aff), os.path.join(img_dir, f"{i:04d}_func_seed_T.nii"))
all_ids = ids + [1099, 5099]
n = len(all_ids)
diag = [1 if i < 4000 else 2 for i in all_ids]
mir = rng.integers(1, 8, n).astype(float)
pd.DataFrame({"VHIS_ID": all_ids, "diagnosis": diag, "There really are miracles?": mir}) \
  .to_excel(os.path.join(dat_dir, "VHIS_Miracles.xlsx"), index=False)
magic = {f"magic_{k}": rng.integers(1, 3, n) for k in range(1, 18)}
pd.DataFrame({"id_nih": all_ids, "fund ave": rng.normal(3, 1, n), "diagnosis": diag, **magic}) \
  .to_excel(os.path.join(dat_dir, "magical_ideation.xlsx"), index=False)
hope = {f"belief_{k}_hope_{k}": rng.integers(1, 8, n).astype(float) for k in range(1, 7)}
hope["belief_2_hope_2"][3] = np.nan
pd.DataFrame({"id_nih": all_ids, **hope}).to_excel(os.path.join(dat_dir, "Dataset_Warfighter_Religious.xlsx"),
                                                   sheet_name="data", index=False)
m = pd.DataFrame({"subject_id": ids, "miracles_score": mir[:len(ids)],
                  "magical_ideation_total": sum(magic.values())[:len(ids)],
                  "network_file_full_path": [os.path.join(img_dir, f"{i:04d}_func_seed_T.nii") for i in ids]})
m.to_excel(os.path.join(dat_dir, "miracles_magical_network_matched.xlsx"), sheet_name="matched", index=False)
print(f"wrote {len(ids)} subjects to {root}")
