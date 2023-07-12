#!/bin/bash

# Help information.
if [[ $# -lt 4 || ${*: -1} == "-h" || ${*: -1} == "--help" ]]; then
    echo "This script tests metrics defined in \`./metrics/\`."
    echo
    echo "Usage: $0 GPUS DATASET MODEL METRICS"
    echo
    echo "Note: More than one metric should be separated by comma." \
         "Also, all metrics assume using all samples from the real dataset" \
         "and 50000 fake samples for GAN-related metrics."
    echo
    echo "Example: $0 1 ~/data/ffhq1024.zip ~/checkpoints/ffhq1024.pth" \
         "fid,is,kid,gan_pr,snapshot,equivariance"
    echo
    exit 0
fi

# Get an available port for launching distributed training.
# Credit to https://superuser.com/a/1293762.
export DEFAULT_FREE_PORT
DEFAULT_FREE_PORT=$(comm -23 <(seq 49152 65535 | sort) \
                    <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) \
                    | shuf | head -n 1)

GPUS=$1
DATASET=$2
MODEL=$3
FNUM=$4
PORT=${PORT:-$DEFAULT_FREE_PORT}

# Parse metrics to test.
METRICS=$5
TEST_FID="false"
TEST_IS="false"
TEST_KID="false"
TEST_GAN_PR="false"
TEST_SNAPSHOT="false"
TEST_EQUIVARIANCE="false"
if [[ ${METRICS} == "all" ]]; then
    TEST_FID="true"
    TEST_IS="true"
    TEST_KID="true"
    TEST_GAN_PR="true"
    TEST_SNAPSHOT="true"
    TEST_EQUIVARIANCE="true"
else
    array=(${METRICS//,/ })
    for var in ${array[@]}; do
        if [[ ${var} == "fid" ]]; then
            TEST_FID="true"
        fi
        if [[ ${var} == "is" ]]; then
            TEST_IS="true"
        fi
        if [[ ${var} == "kid" ]]; then
            TEST_KID="true"
        fi
        if [[ ${var} == "gan_pr" ]]; then
            TEST_GAN_PR="true"
        fi
        if [[ ${var} == "snapshot" ]]; then
            TEST_SNAPSHOT="true"
        fi
        if [[ ${var} == "equivariance" ]]; then
            TEST_EQUIVARIANCE="true"
        fi
    done
fi

# Detect `python3` command.
# This workaround addresses a common issue:
#   `python` points to python2, which is deprecated.
export PYTHONS
export RVAL

PYTHONS=$(compgen -c | grep "^python3$")

# `$?` is a built-in variable in bash, which is the exit status of the most
# recently-executed command; by convention, 0 means success and anything else
# indicates failure.
RVAL=$?

if [ $RVAL -eq 0 ]; then  # if `python3` exist
    PYTHON="python3"
else
    PYTHON="python"
fi

${PYTHON} -m torch.distributed.launch \
    --nproc_per_node=${GPUS} \
    --master_port=${PORT} \
    ./test_metrics.py \
        --launcher="pytorch" \
        --backend="nccl" \
        --dataset ${DATASET} \
        --model ${MODEL} \
        --real_num -1 \
        --fake_num ${FNUM} \
        --test_fid ${TEST_FID} \
        --test_is ${TEST_IS} \
        --test_kid ${TEST_KID} \
        --test_gan_pr ${TEST_GAN_PR} \
        --test_snapshot ${TEST_SNAPSHOT} \
        --test_equivariance ${TEST_EQUIVARIANCE} \
        ${@:6}
