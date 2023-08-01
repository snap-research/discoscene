#!/bin/bash

# Help message.
if [[ $# -lt 2 ]]; then
    echo "This script launches a job of training StyleGAN2-ADA on AFHQ-512."
    echo
    echo "Note: All settings are already preset for training with 8 GPUs." \
         "Please pass addition options, which will overwrite the original" \
         "settings, if needed."
    echo
    echo "Usage: $0 GPUS DATASET [OPTIONS]"
    echo
    echo "Example: $0 8 /data/afhq512.zip [--help]"
    echo
    exit 0
fi

GPUS=$1
DATASET=$2

./scripts/dist_train.sh ${GPUS} stylegan2 \
    --job_name='stylegan2ada_afhq512' \
    --seed=0 \
    --resolution=512 \
    --image_channels=3 \
    --train_dataset=${DATASET} \
    --val_dataset=${DATASET} \
    --val_max_samples=-1 \
    --total_img=25_000_000 \
    --batch_size=8 \
    --val_batch_size=8 \
    --train_data_mirror=true \
    --data_loader_type='iter' \
    --data_repeat=2000 \
    --data_workers=3 \
    --data_prefetch_factor=2 \
    --data_pin_memory=true \
    --g_init_res=4 \
    --latent_dim=512 \
    --d_fmaps_factor=1.0 \
    --g_fmaps_factor=1.0 \
    --d_mbstd_groups=8 \
    --g_num_mappings=8 \
    --d_lr=0.0025 \
    --g_lr=0.0025 \
    --w_moving_decay=0.995 \
    --sync_w_avg=false \
    --style_mixing_prob=0.9 \
    --r1_interval=16 \
    --r1_gamma=0.5 \
    --pl_interval=4 \
    --pl_weight=2.0 \
    --pl_decay=0.01 \
    --pl_batch_shrink=2 \
    --g_ema_img=20_000 \
    --g_ema_rampup=0.0 \
    --eval_at_start=true \
    --eval_interval=3200 \
    --ckpt_interval=3200 \
    --log_interval=64 \
    --enable_amp=false \
    --use_ada=true \
    --num_fp16_res=4 \
    ${@:3}
