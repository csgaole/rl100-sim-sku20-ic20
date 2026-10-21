#!/bin/bash
set -e
S=/data02/kemove/sku09-bc-migration/sku09-flip-demonstrations; W=/data02/kemove/sku5-carton-probe; P=/data02/kemove/sku09-bc-migration/env/bin/python
OUTR=/data02/kemove/sku5-flip-demonstrations/data/v1
declare -A RUN=([SKU5A]=SKU5A_support_scale_a1.0_r1 [SKU5B]=SKU5B_face_offset_a1.0_r2s2 [SKU5D]=SKU5D_face_offset_a1.0_r1 [SKU5E]=SKU5E_face_offset_a1.0_r1)
declare -A SB=([SKU5A]=2000000 [SKU5B]=2000100 [SKU5D]=2000300 [SKU5E]=2000400)
declare -A ID=([SKU5A]=A [SKU5B]=B [SKU5D]=D [SKU5E]=E)
for k in SKU5A SKU5B SKU5D SKU5E; do
  d=$W/strategy/${RUN[$k]}
  for f in $S/assets/ref_$k.npy $S/assets/tcp_jacobian_maps_$k.npy $S/assets/ref_${k}_angles.npy $S/assets/ref_$k.json $OUTR/sku${ID[$k]}; do
    [ -e $f ] && { echo "REFUSE existing $f"; exit 1; }; done
  cp -n $d/baked_ref.npy $S/assets/ref_$k.npy; cp -n $d/baked_maps.npy $S/assets/tcp_jacobian_maps_$k.npy; cp -n $d/baked_angles.npy $S/assets/ref_${k}_angles.npy
  $P -c "import json;b=json.load(open('$d/bake_check.json'));s=json.load(open('$d/summary.json'))['summary'];json.dump(dict(sku='$k',source='CEM strategy: base '+s['base']+' retarget of assets/reference_actions.npy (SKU09) + per-arm Cartesian residual keyframes + timing; baked as effective joint targets of the nominal run',strategy_dir='$d',probe_summary=s,bake_check=b),open('$S/assets/ref_$k.json','w'),indent=1)"
  echo installed $k
done
cd $S
for k in SKU5A SKU5B SKU5D SKU5E; do
  CUDA_VISIBLE_DEVICES= $P src/collect_sku.py search --sku $k --ref assets/ref_$k.npy --maps assets/tcp_jacobian_maps_$k.npy --angles assets/ref_${k}_angles.npy --out $OUTR/sku${ID[$k]} --seed_base ${SB[$k]} --start 0 --stop 100 --workers 32 > $W/logs/collect_search_$k.log 2>&1 &
done; wait; echo search_done $(date)
for k in SKU5A SKU5B SKU5D SKU5E; do
  CUDA_VISIBLE_DEVICES=7 MUJOCO_EGL_DEVICE_ID=7 MUJOCO_GL=egl $P src/collect_sku.py render --sku $k --ref assets/ref_$k.npy --maps assets/tcp_jacobian_maps_$k.npy --angles assets/ref_${k}_angles.npy --out $OUTR/sku${ID[$k]} --seed_base ${SB[$k]} --start 0 --stop 100 --workers 8 > $W/logs/collect_render_$k.log 2>&1 &
done; sleep 8; nvidia-smi --id=7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader; wait; echo render_done $(date)
