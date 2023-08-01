#!/bin/bash

# Help message.
if [[ $# -lt 2 ]]; then
    echo "This script launches a job of training StyleGAN2-ADA on CIFAR10."
    echo
    echo "Note: All settings are already preset for training with 2 GPUs." \
         "Please pass addition options, which will overwrite the original" \
         "settings, if needed."
    echo
    echo "Usage: $0 GPUS DATASET [OPTIONS]"
    echo
    echo "Example: $0 2 /data/cifar10.zip [--help]"
    echo
    exit 0
fi

GPUS=$1
DATASET=$2

./scripts/dist_train.sh ${GPUS} stylegan2 \
    --job_name='stylegan2ada_cifar10' \
    --seed=0 \
    --resolution=32 \
    --image_channels=3 \
    --train_dataset=${DATASET} \
    --train_anno_meta='annotation.json' \
    --val_dataset=${DATASET} \
    --val_anno_meta='annotation.json' \
    --val_max_samples=-1 \
    --total_img=100_000_000 \
    --batch_size=32 \
    --val_batch_size=128 \
    --train_data_mirror=false \
    --data_loader_type='iter' \
    --data_repeat=500 \
    --data_workers=3 \
    --data_prefetch_factor=2 \
    --data_pin_memory=true \
    --g_init_res=4 \
    --latent_dim=512 \
    --label_dim=10 \
    --d_fmaps_factor=1.0 \
    --g_fmaps_factor=1.0 \
    --d_mbstd_groups=8 \
    --g_num_mappings=2 \
    --g_architecture='origin' \
    --d_lr=0.0025 \
    --g_lr=0.0025 \
    --w_moving_decay=0.995 \
    --sync_w_avg=false \
    --style_mixing_prob=0.0 \
    --r1_interval=16 \
    --r1_gamma=0.01 \
    --pl_interval=4 \
    --pl_weight=0.0 \
    --pl_decay=0.0 \
    --pl_batch_shrink=2 \
    --g_ema_img=500_000 \
    --g_ema_rampup=0.05 \
    --eval_at_start=true \
    --eval_interval=3200 \
    --ckpt_interval=3200 \
    --log_interval=64 \
    --enable_amp=false \
    --use_ada=true \
    --num_fp16_res=4 \
    ${@:3}
