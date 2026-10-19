#!/usr/bin/env bash
# Runs every LNM script end to end on synthetic data. Usage: bash tests/run_smoke_test.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
python "$REPO/tests/make_synthetic_data.py" "$WORK"
export VHIS_IMAGE_DIR="$WORK/images" VHIS_DATA_DIR="$WORK/data" VHIS_OUTPUT_DIR="$WORK/outputs"
mkdir -p "$VHIS_OUTPUT_DIR"
for s in 01 02 03 04 05 06 07; do
  echo "== $s"; python "$REPO"/lnm/${s}_*.py > "$WORK/log_$s.txt"
done
echo "== 08"; python "$REPO/lnm/08_miracles_vs_magical_contrast.py" \
  --xlsx "$VHIS_DATA_DIR/miracles_magical_network_matched.xlsx" \
  --outdir "$VHIS_OUTPUT_DIR/miracles_vs_magical_contrast" > "$WORK/log_08.txt"
find "$VHIS_OUTPUT_DIR" -name "*.nii.gz" | wc -l | xargs echo "NIfTI outputs written:"
echo "All scripts ran. Outputs in $WORK"
