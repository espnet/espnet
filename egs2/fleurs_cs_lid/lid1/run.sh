#!/usr/bin/env bash
set -e
set -u
set -o pipefail

# Language identification of utterances with one or two languages.
#   head: softmax_hard (one class per language or atomic language pair),
#         softmax_soft (soft targets, KL loss), sigmoid (multi-hot targets, BCE loss)
#   data: fleurs | fleurs_csfleurs | fleurs_csfleurs_csyodas (training data only: the
#         configs are set for fleurs_csfleurs, see README.md for the other values)
# Stage 7 of lid.sh scores the exact language-set accuracy with local/score.py;
# the union of the four CS-FLEURS test sets is scored after lid.sh.
head=sigmoid
data=fleurs_csfleurs
lid_config=             # default: conf/tuning/train_mms_ecapa_${head}.yaml
lid_tag=
inference_model=valid.accuracy.best.pth
inference_batch_size=4
lid_args=               # options passed to the training (lid.sh --lid_args)
lid_opts=               # other options passed to lid.sh, e.g. "--skip_packing false"
ngpu=1
nj=32
stage=1
stop_stage=10

. utils/parse_options.sh

# Training view and feature type used for the published models
case "${data}" in
    fleurs) view=fleurs; feats_type=raw ;;
    fleurs_csfleurs) view=cs; feats_type=raw ;;
    fleurs_csfleurs_csyodas) view=csyodas; feats_type=raw_copy ;;
    *) echo "Error: unknown --data ${data}" >&2; exit 1 ;;
esac
case "${head}" in
    softmax_hard) lid_label_file=utt2lang; pair=_pair ;;
    softmax_soft) lid_label_file=utt2langs; pair=; feats_type=raw_copy ;;
    sigmoid) lid_label_file=utt2langs; pair= ;;
    *) echo "Error: unknown --head ${head}" >&2; exit 1 ;;
esac
if [ -z "${lid_config}" ]; then
    lid_config=conf/tuning/train_mms_ecapa_${head}.yaml
    if [ "${data}" != fleurs_csfleurs ] && [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
        echo "Error: ${lid_config} is set for --data fleurs_csfleurs;" \
            "give --lid_config with the values in README.md" >&2
        exit 1
    fi
fi
[ -f "${lid_config}" ] || { echo "Error: ${lid_config} not found" >&2; exit 1; }
[ -n "${lid_tag}" ] || lid_tag=$(basename "${lid_config}" .yaml)_${data}_${feats_type}

# Data directories are <split>_fleurscs_<view>[_pair] (local/prepare_fleurs_cs_lid_data.py);
# FLEURS utterances are single-language, so the FLEURS-only sets have no pair view.
if [ "${view}" = fleurs ]; then
    train_set=train_fleurscs_fleurs
    valid_set=valid_fleurscs_fleurs
else
    train_set=train_fleurscs_${view}${pair}
    valid_set=valid_fleurscs_${view}${pair}
fi
cs_test_sets=
for subset in cs_read cs_xtts1 cs_xtts2 cs_mms; do
    cs_test_sets+=" test_fleurscs_${subset}${pair}"
done
test_sets="test_fleurscs_fleurs${cs_test_sets}"

# shellcheck disable=SC2086
./lid.sh \
    --stage "${stage}" --stop_stage "${stop_stage}" \
    --ngpu "${ngpu}" --nj "${nj}" --feats_type "${feats_type}" \
    --train_set "${train_set}" --valid_set "${valid_set}" --test_sets "${test_sets}" \
    --tsne_set "${valid_set}" --extract_embd true --checkpoint_interval 1000 \
    --lid_config "${lid_config}" --lid_tag "${lid_tag}" --lid_args "${lid_args}" \
    --lid_label_file "${lid_label_file}" --lid_stats_dir "exp/lid_stats_${feats_type}_${view}" \
    --inference_model "${inference_model}" --inference_batch_size "${inference_batch_size}" \
    ${lid_opts}

if [ "${stage}" -le 7 ] && [ "${stop_stage}" -ge 7 ]; then
    # Union of the CS-FLEURS test sets, from the per-utterance results of stage 7
    infer_dir=exp/lid_${lid_tag}/inference/${inference_model%.pth}
    details=()
    for dset in ${cs_test_sets}; do
        details+=("${infer_dir}/${dset}/lang_set_details")
    done
    python3 local/score.py --details "${details[@]}" \
        --results "${infer_dir}/test_fleurscs_cs_all${pair}/results"
fi
