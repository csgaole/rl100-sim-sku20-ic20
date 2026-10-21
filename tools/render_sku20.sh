#!/bin/bash
# usage: render_sku20.sh SKU5X GPU   (renders the selected search configs into H5 episodes; EGL on GPU)
set -e
R=/data02/kemove/rl100-sim-sku20-ic20; P=/data02/kemove/sku09-bc-migration/env/bin/python; cd $R; k=$1; G=$2; id=${k#SKU5}
SB=$($P -c "import json;print(json.load(open('assets/ref_$k.json'))['seed_base'])")
CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G MUJOCO_GL=egl $P expert/collect_sku.py render --sku $k --ref assets/ref_$k.npy --maps assets/tcp_jacobian_maps_$k.npy --angles assets/ref_${k}_angles.npy --out data/v1/sku$id --seed_base $SB --start 0 --stop 100 --workers 8 > logs/collect_render_$k.log 2>&1
echo render_done $k $(date)
