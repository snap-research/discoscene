#!/bin/bash

# Help message.
if [[ $# -lt 2 ]]; then
    echo "This script launches a job of training StyleGAN on FFHQ-256."
    echo
    echo "Note: All settings are already preset for training with 8 GPUs." \
         "Please pass addition options, which will overwrite the original" \
         "settings, if needed."
    echo
    echo "Usage: $0 GPUS DATASET [OPTIONS]"
    echo
    echo "Example: $0 8 /data/ffhq256.zip [--help]"
    echo
    exit 0
fi

GPUS=$1
DATASET=$2
DATASETANN=$3
PORT=${PORT:-1234}

PORT=$PORT ./scripts/dist_train.sh ${GPUS} discoscene \
    --job_name='discoscene_res256' \
    --seed=0 \
    --resolution=256 \
    --image_channels=3 \
    --train_dataset=${DATASET} \
    --val_dataset=${DATASET} \
    --train_anno_path=${DATASETANN} \
    --val_anno_path=${DATASETANN} \
    --val_max_samples=-1 \
    --total_img=25_000_000 \
    --batch_size=8 \
    --val_batch_size=16 \
    --train_data_mirror=true \
    --data_loader_type='iter' \
    --data_repeat=200 \
    --data_workers=3 \
    --data_prefetch_factor=2 \
    --data_pin_memory=true \
    --w_moving_decay=0.995 \
    --sync_w_avg=false \
    --r1_gamma=0.2 \
    --g_ema_img=10_000 \
    --eval_at_start=true \
    --eval_interval=6400 \
    --ckpt_interval=6400 \
    --log_interval=128 \
    --use_ada=false \
    --enable_amp=false \
    --g_fmaps_factor=0.25 \
    --logger_type=normal \
    --log_interval=50 \
    ${@:4}
