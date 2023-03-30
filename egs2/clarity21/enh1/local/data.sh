#!/usr/bin/env bash

set -e
set -u
set -o pipefail

help_message=$(cat << EOF
Usage: $0 --clarity_root <path> [--sample_rate <sample_rate>]

  required argument:
    --clarity_root: path to clarity dataset root folder. i.e. folder which contains train and dev subfolders.
    NOTE:


  optional argument:
    [--sample_rate]: 16000 (default) or 44100
EOF
)

clarity_root=
sample_rate=16000

. utils/parse_options.sh

# check for sox
! command -v sox &>/dev/null && echo "sox: command not found" && exit 1;

python local/prep_data.py --clarity_root ${clarity_root} --fs ${sample_rate}
