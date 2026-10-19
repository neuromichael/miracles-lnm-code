# Analysis code: belief in miracles and magical ideation in the VHIS lesion cohort

Code for the lesion network mapping (LNM) and specificity analyses in Cristofori et al., "Distinct neurocognitive signatures of miracle belief and magical ideation" (PNAS, under revision, tracking number 2026-20645).

Participant data are not included. See [Data availability](#data-availability).

## What is here

| Folder | Contents |
|---|---|
| `lnm/` | Voxelwise lesion network mapping, spatial correlations, permutation inference and the miracles versus magical ideation contrast |
| `vlsm/` | Voxel-based lesion-symptom mapping (to be added, see `vlsm/README.md`) |
| `data/` | Description of the expected input files. No data. |
| `tests/` | Synthetic data generator and an end-to-end smoke test |

The scripts in `lnm/` are the scripts that produced the reported results. The only change from the versions that were run is that hardcoded file paths were replaced by the environment variables described below.

## Lesion network mapping pipeline

Each subject has a lesion mask (`<ID>.nii`) and a lesion network map (`<ID>_func_seed_T.nii`): the voxelwise t statistic of functional connectivity between the lesion and every brain voxel, computed in the GSP1000 normative connectome (n = 1,000). All images are in 2 mm MNI space (91 x 109 x 91).

TODO: name the software and version used to generate the `_func_seed_T` maps and the connectome preprocessing.

At every voxel v, each script fits an ordinary least squares model across subjects i:

```
N_i(v) = b0(v) + b_score(v) * score_i + b_vol(v) * lesion_volume_i + e_i(v)
```

The output is the unthresholded t map for the score regressor. Lesion volume (mm³) is computed from the lesion mask and entered as a nuisance covariate. The analysis mask is the set of voxels that are finite and nonzero in every subject's network map. A positive t means stronger lesion connectivity to that voxel goes with a higher score.

| Script | Analysis |
|---|---|
| `01_miracles_voxelwise_lnm.py` | Belief in miracles ("There really are miracles", 1 to 7) |
| `02_magical_ideation_items_voxelwise_lnm.py` | One map per Magical Ideation item, items as coded |
| `03_magical_ideation_items_voxelwise_lnm_reversed.py` | Same, with items 2, 7, 8 and 14 reverse scored per the Eckblad and Chapman (1983) key |
| `04_miracles_vs_magical_items_spatial_correlations.py` | Spatial correlation of the miracles map with each item map; Cronbach's alpha of the items |
| `05_religious_hope_voxelwise_lnm.py` | Religious Hope subscale (items 12 to 17 of Sethi and Seligman, 1993), prorated sum |
| `06_religious_hope_fdr_fwe.py` | FDR (Benjamini-Hochberg) and FWE (Freedman-Lane, 5,000 permutations, max statistic) thresholding of the Hope map |
| `07_religious_hope_items_voxelwise_lnm.py` | One map per Hope item and their pairwise spatial correlations |
| `08_miracles_vs_magical_contrast.py` | Joint model with both scores and the contrast b_miracles minus b_magical; map submitted to the Neurosynth decoder |

Scripts 04 and 06 read the outputs of 01, 03 and 05, so run the scripts in numerical order.

The Neurosynth decoding used the web decoder (https://neurosynth.org/decode/) on the unthresholded contrast map written by script 08.

## Requirements

Python 3.10 or later with the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The code was run with Python 3.13, numpy 2.4, pandas 3.0, nibabel 5.4, scipy 1.17 and openpyxl 3.1.

## Running

Set three environment variables, then run the scripts in order:

```bash
export VHIS_IMAGE_DIR=/path/to/images       # <ID>.nii and <ID>_func_seed_T.nii
export VHIS_DATA_DIR=/path/to/spreadsheets  # see data/README.md
export VHIS_OUTPUT_DIR=/path/to/outputs
for s in lnm/0[1-7]_*.py; do python "$s"; done
python lnm/08_miracles_vs_magical_contrast.py --xlsx "$VHIS_DATA_DIR/miracles_magical_network_matched.xlsx" --outdir "$VHIS_OUTPUT_DIR/miracles_vs_magical_contrast"
```

Each script writes its maps, an `included_subjects.csv` and a timestamped `run_log.txt` to its own folder under `VHIS_OUTPUT_DIR`.

## Testing without the real data

`tests/run_smoke_test.sh` generates random images and spreadsheets with the same structure as the real inputs and runs every script on them:

```bash
bash tests/run_smoke_test.sh
```

The synthetic data only check that the code runs. They carry no information about the cohort.

## Data availability

Individual-level data from the W.F. Caveness Vietnam Head Injury Study cannot be shared publicly because participants did not consent to open release. TODO: describe the access route for qualified researchers (contact, data use agreement, approving committee).

The unthresholded group-level maps reported in the paper (TODO: NeuroVault collection link) contain no individual-level information and can be shared openly.

## License

Code is released under the MIT License (see `LICENSE`).

## Citation

TODO: add the article reference and DOI on publication, and the archived release DOI (for example from Zenodo).
