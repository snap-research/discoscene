#!/bin/bash

# Help message.
if [[ $# -lt 2 ]]; then
    echo "This script launches a job of training StyleGAN3 (config R) on" \
         "FFHQ-U-1024."
    echo
    echo "Note: All settings are already preset for training with 8 GPUs." \
         "Please pass addition options, which will overwrite the original" \
         "settings, if needed."
    echo
    echo "Usage: $0 GPUS DATASET [OPTIONS]"
    echo
    echo "Example: $0 8 /data/ffhqu1024.zip [--help]"
    echo
    exit 0
fi

GPUS=$1
DATASET=$2

./scripts/dist_train.sh ${GPUS} stylegan3 \
    --job_name='stylegan3r_ffhqu1024' \
    --seed=0 \
    --resolution=1024 \
    --image_channels=3 \
    --train_dataset=${DATASET} \
    --val_dataset=${DATASET} \
    --val_max_samples=-1 \
    --total_img=25_000_000 \
    --batch_size=4 \
    --val_batch_size=4 \
    --train_data_mirror=true \
    --data_loader_type='iter' \
    --data_repeat=200 \
    --data_workers=3 \
    --data_prefetch_factor=2 \
    --data_pin_memory=true \
    --g_kernel_size=1 \
    --latent_dim=512 \
    --d_fmaps_factor=1.0 \
    --g_fmaps_factor=1.0 \
    --d_mbstd_groups=4 \
    --g_num_mappings=2 \
    --d_lr=0.002 \
    --g_lr=0.0025 \
    --w_moving_decay=0.998 \
    --sync_w_avg=false \
    --style_mixing_prob=0.0 \
    --r1_interval=16 \
    --r1_gamma=32.8 \
    --blur_init_sigma=10.0 \
    --blur_fade_img=200_000 \
    --pl_interval=0 \
    --pl_weight=0.0 \
    --pl_decay=0.01 \
    --pl_batch_shrink=2 \
    --g_ema_img=10_000 \
    --g_ema_rampup=0.05 \
    --eval_at_start=true \
    --eval_interval=6400 \
    --ckpt_interval=6400 \
    --log_interval=128 \
    --enable_amp=false \
    --use_ada=true \
    --num_fp16_res=4 \
    ${@:3}
