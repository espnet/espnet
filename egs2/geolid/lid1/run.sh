#!/usr/bin/env bash
set -euo pipefail

setup=voxlingua107  # voxlingua107 or combined
stage=1
stop_stage=10000
ngpu=1
nj=8
python=python3
local_data_opts=
lid_config=  # Defaults to the selected setup's training configuration.

help_message="Usage: $0 [--setup voxlingua107|combined] [--stage N] [--stop_stage N]
Options: --ngpu, --nj, --python, --local_data_opts, --lid_config.
Combined preparation runs in stages 1-3; resume training at stage 4.
Edit the explicit lid.sh options below for other pipeline settings."
. utils/parse_options.sh
if [ $# -ne 0 ]; then
    echo "${help_message}" >&2
    exit 2
fi
if ! [[ "${stage}" =~ ^[0-9]+$ && "${stop_stage}" =~ ^[0-9]+$ ]]; then
    echo "$0: stage and stop_stage must be non-negative integers" >&2
    exit 2
fi

case "${setup}" in
    voxlingua107)
        train_set=train_voxlingua107_lang
        valid_set=dev_voxlingua107_lang
        test_sets=dev_voxlingua107_lang
        lid_config=${lid_config:-conf/train.yaml}
        ;;
    combined)
        train_set=train_all_no_filter_lang
        valid_set=dev_ml_superb2_lang
        test_sets="dev_voxlingua107_lang test_voxpopuli_lang test_fleurs_lang dev_ml_superb2_lang dev_dialect_ml_superb2_lang dev_babel_over_10s_lang"
        lid_config=${lid_config:-conf/tuning/combined.yaml}
        if [ "${stage}" -le 3 ]; then
            local/prepare_combined.sh \
                --stage "${stage}" --stop_stage "${stop_stage}" \
                --nj "${nj}" --python "${python}" \
                --local_data_opts "${local_data_opts}"
            if [ "${stop_stage}" -le 3 ]; then
                exit 0
            fi
            stage=4
        fi
        ;;
    *) echo "$0: unknown setup '${setup}' (choose voxlingua107 or combined)" >&2; exit 2 ;;
esac

./lid.sh \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --ngpu "${ngpu}" \
    --nj "${nj}" \
    --python "${python}" \
    --local_data_opts "${local_data_opts}" \
    --feats_type raw \
    --fs 16k \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --tsne_set "${valid_set}" \
    --lid_config "${lid_config}" \
    --expdir "exp_${setup}" \
    --inference_model valid.accuracy.best.pth \
    --inference_batch_size 1 \
    --extract_embd true \
    --checkpoint_interval 1000 \
    --max_utt_per_lang_for_tsne 100
