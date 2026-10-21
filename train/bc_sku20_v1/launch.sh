#!/bin/bash
# SKU20 BC launcher — IDENTICAL hyper-parameters to rl100-sim-repro/train/bc_multisku_v1_20261018/launch.sh as used for s43_lr2e-4_mb256
# (seed 43, lr 2e-4, microbatch 256, effective batch 512, fp32, 400 epochs, first gate 100, gate every 50, min stop 200, 10 eval procs/GPU).
# NOT LAUNCHED. Usage (after Bill confirms):  bash launch.sh <GPU> s43_lr2e-4_mb256 43 2e-4 256
# Prereq: cache built once:  python build_cache_multisku.py --out /data02/kemove/rl100-sim-sku20-ic20/cache/bc_v1
cd /data02/kemove/rl100-sim-sku20-ic20/train/bc_sku20_v1; G=$1; N=$2; R=/data02/kemove/rl100-sim-sku20-ic20/runs/bc_sku20_v1
mkdir -p $R; [ -e $R/$N ] && { echo "refuse: $R/$N exists"; exit 1; }
used=$(nvidia-smi --id=$G --query-gpu=memory.used --format=csv,noheader,nounits); [ "$used" -lt 1000 ] || { echo "refuse: GPU $G busy ($used MiB) - never displace other users"; exit 1; }
CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G MUJOCO_GL=egl P3_TRAIN_GPU=$G nohup /data02/kemove/sku09-bc-migration/env/bin/python -u train_bc_multisku.py --run-dir $R/$N --mirror-dir $R/${N}_mirror --seed $3 --lr $4 --eval-gpus $G --eval-procs-per-gpu 10 --max-epochs 400 --first-gate-epoch 100 --gate-every 50 --min-stop-epoch 200 --microbatch $5 --effective-batch 512 --precision fp32 > $R/$N.log 2>&1 &
echo $! > $R/$N.pid; echo $N pid $! gpu $G
