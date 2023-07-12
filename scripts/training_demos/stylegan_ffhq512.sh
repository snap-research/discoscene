#!/bin/bash

# Help message.
if [[ $# -lt 2 ]]; then
    echo "This script launches a job of training StyleGAN on FFHQ-512."
    echo
    echo "Note: All settings are already preset for training with 8 GPUs." \
         "Please pass addition options, which will overwrite the original" \
         "settings, if needed."
    echo
    echo "Usage: $0 GPUS DATASET [OPTIONS]"
    echo
    echo "Example: $0 8 /data/ffhq512.zip [--help]"
    echo
    exit 0
fi

GPUS=$1
DATASET=$2

./scripts/dist_train.sh ${GPUS} stylegan \
    --job_name='stylegan_ffhq512' \
    --seed=0 \
    --resolution=512 \
    --image_channels=3 \
    --train_dataset=${DATASET} \
    --val_dataset=${DATASET} \
    --val_max_samples=-1 \
    --total_img=25_000_000 \
    --batch_size=4 \
    --val_batch_size=8 \
    --train_data_mirror=true \
    --data_loader_type='iter' \
    --data_repeat=200 \
    --data_workers=3 \
    --data_prefetch_factor=2 \
    --data_pin_memory=true \
    --g_init_res=4 \
    --latent_dim=512 \
    --d_fmaps_factor=1.0 \
    --g_fmaps_factor=1.0 \
    --d_mbstd_groups=4 \
    --g_num_mappings=8 \
    --d_lr=0.001 \
    --g_lr=0.001 \
    --w_moving_decay=0.995 \
    --sync_w_avg=false \
    --style_mixing_prob=0.9 \
    --r1_gamma=10.0 \
    --g_ema_img=10_000 \
    --eval_at_start=true \
    --eval_interval=6400 \
    --ckpt_interval=6400 \
    --log_interval=128 \
    --use_ada=false \
    --enable_amp=true \
    -o controllers.ProgressScheduler.init_res=8 \
    -o controllers.ProgressScheduler.batch_size_schedule='{"res4":64,"res8":32,"res16":16,"res32":8}' \
    -o controllers.ProgressScheduler.lr_schedule='{"res128":1.5,"res256":2,"res512":3,"res1024":3}' \
    ${@:3}
