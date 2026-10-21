#!/bin/bash
# SKU20 BC DDP launcher: identical hparams to launch.sh, global batch 512 = NGPU x microbatch x accum (eval gate on first GPU).
# Usage: bash launch_ddp.sh <GPU list e.g. 0,1,2,3,4,5,6,7> <run name> <seed> <lr> <per-GPU microbatch> [extra args]
cd /data02/kemove/rl100-sim-sku20/train/bc_sku20_v1; G=$1; N=$2; SEED=$3; LR=$4; MB=$5; R=/data02/kemove/rl100-sim-sku20/runs/bc_sku20_v1; shift 5; EXTRA="$@"
NG=$(echo $G | tr ',' '\n' | wc -l); G0=${G%%,*}
mkdir -p $R; [ -e $R/$N ] && { echo "refuse: $R/$N exists"; exit 1; }
CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G0 MUJOCO_GL=egl P3_TRAIN_GPU=$G0 nohup /data02/kemove/sku09-bc-migration/env/bin/python -m torch.distributed.run --standalone --nproc_per_node=$NG train_bc_multisku.py --run-dir $R/$N --mirror-dir $R/${N}_mirror --seed $SEED --lr $LR --eval-gpus $G0 --eval-procs-per-gpu 10 --max-epochs 400 --first-gate-epoch 100 --gate-every 50 --min-stop-epoch 200 --microbatch $MB --effective-batch 512 --precision fp32 $EXTRA > $R/$N.log 2>&1 &
echo $! > $R/$N.pid; echo $N pid $! gpus $G
