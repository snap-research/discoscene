#!/bin/bash

# Help information.
if [[ $# -lt 2 || ${*: -1} == "-h" || ${*: -1} == "--help" ]]; then
    echo "This script tests model converters defined in \`./converters/\`."
    echo
    echo "Usage: $0 CKPT_DIR TEST_MODEL_TYPE [ADDITIONAL_ARGS]"
    echo
    echo "Note: More than one model type should be separated by comma."
    echo
    echo "Example: $0 ~/checkpoints stylegan2,stylegan3 [--save_test_image]"
    echo
    exit 0
fi

# Get checkpoint directory.
CKPT_DIR=$1

# Parse model types to test.
TEST_MODEL_TYPE=$2
TEST_PGGAN="false"
TEST_STYLEGAN="false"
TEST_STYLEGAN2="false"
TEST_STYLEGAN2ADA_TF="false"
TEST_STYLEGAN2ADA_PTH="false"
TEST_STYLEGAN3="false"
if [[ ${TEST_MODEL_TYPE} == "all" ]]; then
    TEST_PGGAN="true"
    TEST_STYLEGAN="true"
    TEST_STYLEGAN2="true"
    TEST_STYLEGAN2ADA_TF="true"
    TEST_STYLEGAN2ADA_PTH="true"
    TEST_STYLEGAN3="true"
else
    array=(${TEST_MODEL_TYPE//,/ })
    for var in ${array[@]}; do
        if [[ ${var} == "pggan" ]]; then
            TEST_PGGAN="true"
        fi
        if [[ ${var} == "stylegan" ]]; then
            TEST_STYLEGAN="true"
        fi
        if [[ ${var} == "stylegan2" ]]; then
            TEST_STYLEGAN2="true"
        fi
        if [[ ${var} == "stylegan2ada_tf" ]]; then
            TEST_STYLEGAN2ADA_TF="true"
        fi
        if [[ ${var} == "stylegan2ada_pth" ]]; then
            TEST_STYLEGAN2ADA_PTH="true"
        fi
        if [[ ${var} == "stylegan3" ]]; then
            TEST_STYLEGAN3="true"
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

# Test converters if needed.
if [[ ${TEST_PGGAN} == "true" ]]; then
    echo "=========================================="
    echo "=          Test PGGAN Converter          ="
    echo "=========================================="
    echo
    rm ${CKPT_DIR}/pggan_*.pth
    for name in $(ls ${CKPT_DIR}/pggan_*.pkl); do
        ${PYTHON} convert_model.py pggan \
                                   --source_model_path ${name} \
                                   --target_model_path ${name//.pkl/.pth} \
                                   ${@:3}
    done
    echo
fi
if [[ ${TEST_STYLEGAN} == "true" ]]; then
    echo "=========================================="
    echo "=         Test StyleGAN Converter        ="
    echo "=========================================="
    echo
    rm ${CKPT_DIR}/stylegan_*.pth
    for name in $(ls ${CKPT_DIR}/stylegan_*.pkl); do
        ${PYTHON} convert_model.py stylegan \
                                   --source_model_path ${name} \
                                   --target_model_path ${name//.pkl/.pth} \
                                   ${@:3}
    done
    echo
fi
if [[ ${TEST_STYLEGAN2} == "true" ]]; then
    echo "=========================================="
    echo "=        Test StyleGAN2 Converter        ="
    echo "=========================================="
    echo
    rm ${CKPT_DIR}/stylegan2_*.pth
    for name in $(ls ${CKPT_DIR}/stylegan2_*.pkl); do
        ${PYTHON} convert_model.py stylegan2 \
                                   --source_model_path ${name} \
                                   --target_model_path ${name//.pkl/.pth} \
                                   ${@:3}
    done
    echo
fi
if [[ ${TEST_STYLEGAN2ADA_TF} == "true" ]]; then
    echo "=========================================="
    echo "=     Test StyleGAN2ADA TF Converter     ="
    echo "=========================================="
    echo
    rm ${CKPT_DIR}/stylegan2adaTF_*.pth
    for name in $(ls ${CKPT_DIR}/stylegan2adaTF_*.pkl); do
        ${PYTHON} convert_model.py stylegan2ada_tf \
                                   --source_model_path ${name} \
                                   --target_model_path ${name//.pkl/.pth} \
                                   ${@:3}
    done
    echo
fi
if [[ ${TEST_STYLEGAN2ADA_PTH} == "true" ]]; then
    echo "=========================================="
    echo "=     Test StyleGAN2ADA PTH Converter    ="
    echo "=========================================="
    echo
    rm ${CKPT_DIR}/stylegan2adaPTH_*.pth
    for name in $(ls ${CKPT_DIR}/stylegan2adaPTH_*.pkl); do
        ${PYTHON} convert_model.py stylegan2ada_pth \
                                   --source_model_path ${name} \
                                   --target_model_path ${name//.pkl/.pth} \
                                   ${@:3}
    done
    echo
fi
if [[ ${TEST_STYLEGAN3} == "true" ]]; then
    echo "=========================================="
    echo "=        Test StyleGAN3 Converter        ="
    echo "=========================================="
    echo
    rm ${CKPT_DIR}/stylegan3*.pth
    for name in $(ls ${CKPT_DIR}/stylegan3*.pkl); do
        ${PYTHON} convert_model.py stylegan3 \
                                   --source_model_path ${name} \
                                   --target_model_path ${name//.pkl/.pth} \
                                   ${@:3}
    done
    echo
fi
