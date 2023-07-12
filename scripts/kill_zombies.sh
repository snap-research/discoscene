#!/bin/bash

# Help information.
if [[ $# -le 0 || ${*: -1} == "-h" || ${*: -1} == "--help" ]]; then
    echo "This script kills processes launched with" \
         "\`./scripts/dist_train.sh\`, with arguments as keywords to filter."
    echo
    echo "Note: It does NOT check whether they are zombies. Hence," \
         "to ensure killing the desired processes rather than innocent ones," \
         "you MUST provide sufficient arguments to identified targets."
    echo
    echo "Usage: $0 [any arguments pass to your \`dist_train.sh\`]."
    echo
    echo "Example: $0 configs/stylegan2_config.py --work_dir work_dirs/debug"
    echo
    exit 0
fi

kill -9 echo $(ps ux | grep "$*" | grep -v grep | awk '{print $2}')
