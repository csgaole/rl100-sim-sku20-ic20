#!/bin/bash
# usage: install_search_sku20.sh SKU5X <strategy_run_name>   (bake -> install ref assets -> CPU search 100 demos)
set -e
R=/data02/kemove/rl100-sim-sku20-ic20; P=/data02/kemove/sku09-bc-migration/env/bin/python; cd $R
k=$1; run=$2; id=${k#SKU5}; d=$R/strategy/$run
idx=$($P -c "print('ABCDEFGHIJKLMNOPQRST'.index('$id'))"); SB=$((2000000+100*idx))
[ -f $d/bake_check.json ] || $P tools/bake.py $d > logs/bake_$run.log 2>&1
for f in assets/ref_$k.npy assets/tcp_jacobian_maps_$k.npy assets/ref_${k}_angles.npy assets/ref_$k.json data/v1/sku$id; do [ -e $f ] && { echo "REFUSE existing $f"; exit 1; }; done
cp -n $d/baked_ref.npy assets/ref_$k.npy; cp -n $d/baked_maps.npy assets/tcp_jacobian_maps_$k.npy; cp -n $d/baked_angles.npy assets/ref_${k}_angles.npy
$P -c "import json;b=json.load(open('$d/bake_check.json'));s=json.load(open('$d/summary.json'))['summary'];json.dump(dict(sku='$k',seed_base=$SB,source='CEM strategy: base '+s['base']+' retarget of assets/reference_actions.npy (SKU09) + per-arm Cartesian residual keyframes + timing; baked as effective joint targets of the nominal run (same recipe as SKU5A/B/D/E)',strategy_dir='$d',probe_summary=s,bake_check=b),open('assets/ref_$k.json','w'),indent=1)"
CUDA_VISIBLE_DEVICES= $P expert/collect_sku.py search --sku $k --ref assets/ref_$k.npy --maps assets/tcp_jacobian_maps_$k.npy --angles assets/ref_${k}_angles.npy --out data/v1/sku$id --seed_base $SB --start 0 --stop 100 --workers 12 > logs/collect_search_$k.log 2>&1
echo search_done $k $(date)
