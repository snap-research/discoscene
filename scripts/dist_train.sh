#!/bin/bash

# Detect `python3` command.
# This workaround addresses a common issue:
#   `python` points to `python2`, which is deprecated.
export PYTHONS
export RVAL

PYTHONS=$(compgen -c | grep "^python3$")

# `$?` is a built-in variable in bash, which is the exit status of the most
# recently-executed command; by convention, 0 means success and anything else
# indicates failure.
RVAL=$?

if [[ $RVAL -eq 0 ]]; then  # if `python3` exist
    PYTHON="python3"
else
    PYTHON="python"
fi

# Help message.
if [[ $# -lt 2 ]]; then
    echo "This script helps launch distributed training job on local machine."
    echo
    echo "Usage: $0 GPUS COMMAND [ARGS]"
    echo
    echo "Example: $0 8 stylegan2 [--help]"
    echo
    echo "Detailed instruction on available commands:"
    echo "--------------------------------------------------"
    ${PYTHON} ./train.py --help
    echo
    exit 0
fi

GPUS=$1
COMMAND=$2

# Help message for a particular command.
if [[ $# -lt 3 || ${*: -1} == "--help" ]]; then
    echo "Detailed instruction on the arguments for command \`"${COMMAND}"\`:"
    echo "--------------------------------------------------"
    ${PYTHON} ./train.py ${COMMAND} --help
    echo
    exit 0
fi

# Switch memory allocator if available.
# Search order: jemalloc.so -> tcmalloc.so.
# According to https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html,
# it can get better performance by reusing memory as much as possible than
# default malloc function.
JEMALLOC=$(ldconfig -p | grep -i "libjemalloc.so$" | tr " " "\n" | grep "/" \
           | head -n 1)
TCMALLOC=$(ldconfig -p | grep -i "libtcmalloc.so.4$" | tr " " "\n" | grep "/" \
           | head -n 1)
if [ -n "$JEMALLOC" ]; then  # if found the path to libjemalloc.so
    echo "Switch memory allocator to jemalloc."
    export LD_PRELOAD=$JEMALLOC:$LD_PRELOAD
elif [ -n "$TCMALLOC" ]; then  # if found the path to libtcmalloc.so.4
    echo "Switch memory allocator to tcmalloc."
    export LD_PRELOAD=$TCMALLOC:$LD_PRELOAD
fi

# Get an available port for launching distributed training.
# Credit to https://superuser.com/a/1293762.
export DEFAULT_FREE_PORT
DEFAULT_FREE_PORT=$(comm -23 <(seq 49152 65535 | sort) \
                    <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) \
                    | shuf | head -n 1)

PORT=${PORT:-$DEFAULT_FREE_PORT}

${PYTHON} -m torch.distributed.launch \
    --nproc_per_node=${GPUS} \
    --master_port=${PORT} \
    ./train.py \
        --launcher="pytorch" \
        --backend="nccl" \
        ${COMMAND} \
        ${@:3} \
        || exit 1  # Stop the script when it finds exception threw by Python.
