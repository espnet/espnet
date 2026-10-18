#!/usr/bin/env bash
set -euo pipefail
tuning_config=
python_cmd=python
args=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --tuning_config) tuning_config=$2; shift 2 ;;
        --python) python_cmd=$2; args+=("$1" "$2"); shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done
config=conf/train.yaml
if [ -n "${tuning_config}" ]; then
    . ./path.sh
    config="exp/configs/$(basename "${tuning_config}")"
    ${python_cmd} local/merge_config.py conf/train.yaml "${tuning_config}" "${config}"
fi
exec bash ./run_voxlingua107_only.sh --lid_config "${config}" "${args[@]}"
