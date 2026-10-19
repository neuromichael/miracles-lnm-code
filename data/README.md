# Input data

Participant data are not distributed with this repository. The scripts expect the following files in `VHIS_DATA_DIR`. Column names must match exactly.

## `VHIS_Miracles.xlsx` (first sheet)

| Column | Content |
|---|---|
| 1 | VHIS ID (integer) |
| 2 | Group code (1 or 2; IDs below 4000 are coded 1) |
| 3 | "There really are miracles?" (1 to 7) |

Only the first three columns are read, by position.

## `magical_ideation.xlsx` (first sheet)

`id_nih`, `diagnosis`, `magic_1` to `magic_17` (1 = false, 2 = true). Other columns are ignored.

## `Dataset_Warfighter_Religious.xlsx` (sheet `data`)

`id_nih` and the Religious Hope items `belief_1_hope_1` to `belief_6_hope_6` (1 to 7). The sheet also holds the influence and involvement items of the Sethi and Seligman (1993) measure and the 17 Magical Ideation items.

## `miracles_magical_network_matched.xlsx` (sheet `matched`)

Input to script 08: `subject_id`, `miracles_score`, `magical_ideation_total`, `network_file_full_path`.

## Images (`VHIS_IMAGE_DIR`)

`<ID>.nii` (binary lesion mask) and `<ID>_func_seed_T.nii` (lesion network map) for each subject, with the ID zero-padded to four digits, all in the same 2 mm MNI grid.
