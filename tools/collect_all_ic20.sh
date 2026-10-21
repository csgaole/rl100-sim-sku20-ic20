#!/bin/bash
# 20 SKU x 100 demos; IC p20_y10; DO NOT start until Bill confirms.
set -euo pipefail
R=/data02/kemove/rl100-sim-sku20-ic20
P=/data02/kemove/sku09-bc-migration/env/bin/python
cd "$R"
SKUS=(SKU5A SKU5B SKU5C SKU5D SKU5E SKU5F SKU5G SKU5H SKU5I SKU5J SKU5K SKU5L SKU5M SKU5N SKU5O SKU5P SKU5Q SKU5R SKU5S SKU5T)
MODE=${1:-all}
G=${RENDER_GPU:-4}
search_one() {
  local k=$1 id=${1#SKU5}
  local idx=$($P -c "print('ABCDEFGHIJKLMNOPQRST'.index('$id'))")
  local SB=$((2000000+100*idx))
  echo search $k seed_base=$SB
  CUDA_VISIBLE_DEVICES= $P expert/collect_sku.py search --sku "$k" --ref assets/ref_$k.npy --maps assets/tcp_jacobian_maps_$k.npy --angles assets/ref_${k}_angles.npy --out data/v1/sku$id --seed_base $SB --start 0 --stop 100 --workers 12 > logs/collect_search_${k}_ic20.log 2>&1
}
render_one() {
  local k=$1 id=${1#SKU5} g=$2
  local SB=$($P -c "import json;print(json.load(open('assets/ref_$k.json'))['seed_base'])")
  echo render $k GPU $g
  CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g MUJOCO_GL=egl $P expert/collect_sku.py render --sku "$k" --ref assets/ref_$k.npy --maps assets/tcp_jacobian_maps_$k.npy --angles assets/ref_${k}_angles.npy --out data/v1/sku$id --seed_base $SB --start 0 --stop 100 --workers 8 > logs/collect_render_${k}_ic20.log 2>&1
}
if [[ "$MODE" == search || "$MODE" == all ]]; then for k in "${SKUS[@]}"; do search_one "$k" & done; wait; echo search_done; fi
if [[ "$MODE" == render || "$MODE" == all ]]; then for k in "${SKUS[@]}"; do render_one "$k" "$G"; done; echo render_done; fi
