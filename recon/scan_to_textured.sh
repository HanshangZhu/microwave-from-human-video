#!/usr/bin/env bash
# Scan shots of an articulated object -> its measured URDF with the scanned colours.
#
#   bash recon/scan_to_textured.sh <scan dir> <urdf> <work dir> <shot> [<shot> ...]
#   bash recon/scan_to_textured.sh Gitignore-but-important/laptop/microwave \
#        assets/microwave/microwave.urdf Gitignore-but-important/recon/microwave closed open_90
#
# Per shot: 40 sharpest frames + aligned depth (capture_to_r2s.py), Pi3X poses with the
# real intrinsics + fused RealSense depth (fuse_depth.py --cond K), model registration
# and door angle (register_model.py -> look at registration_overlay.png).
# Then all shots are baked together (bake_colors.py) into <urdf dir>/textured/ and
# <urdf name>_textured.urdf.
set -eo pipefail
SCAN="$1"; URDF="$2"; WORK="$3"; shift 3
[[ $# -ge 1 ]] || { echo "usage: $0 <scan dir> <urdf> <work dir> <shot> [<shot> ...]"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
run(){ PYTHONNOUSERSITE=1 conda run --no-capture-output -n "$1" python "${@:2}"; }

SHOTS=()
for s in "$@"; do
  echo "== $s"
  run rscap "$HERE/capture_to_r2s.py" "$SCAN" "$s" --out "$WORK/$s" --n 40
  run 3dgrut "$HERE/fuse_depth.py" "$WORK/$s" --out "$WORK/$s/run" --cond K
  run 3dgrut "$HERE/register_model.py" "$WORK/$s" "$WORK/$s/run" --urdf "$URDF"
  SHOTS+=(--shot "$WORK/$s" "$WORK/$s/run")
done
echo "== bake"
run 3dgrut "$HERE/bake_colors.py" --urdf "$URDF" "${SHOTS[@]}" --out "$(dirname "$URDF")/textured"
