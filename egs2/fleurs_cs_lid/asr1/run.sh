#!/usr/bin/env bash
set -e
set -u
set -o pipefail

# Encoder-decoder language-set generator: the target is <eng> or <ara> <eng>.
# The default config is phase 1 (MMS-1B frozen); phase 2 is a second run with
# the full fine-tuning config and --pretrained_model (see README.md).
# local/score_lidseq.py then scores the decoded language sets.
data=fleurs_csfleurs    # training data: fleurs | fleurs_csfleurs | fleurs_csfleurs_csyodas
                        # (the configs are set for fleurs_csfleurs, see README.md)
asr_config=conf/tuning/train_mms_transformer_pit_frozen.yaml
pretrained_model=       # model to initialise from (asr.sh --pretrained_model)
inference_config=conf/decode.yaml
inference_asr_model=valid.loss.best.pth
asr_args=               # options passed to the training (asr.sh --asr_args)
asr_opts=               # other options passed to asr.sh, e.g. "--skip_packing false"
ngpu=1
nj=32
inference_nj=4
stage=1
stop_stage=15

. utils/parse_options.sh

# Data view, feature type and duration bound used for the published models;
# the data directories are shared with ../lid1 (local/data.sh).
case "${data}" in
    fleurs) view=fleurs; feats_type=raw; max_wav_duration=30 ;;
    fleurs_csfleurs) view=cs; feats_type=raw; max_wav_duration=30 ;;
    fleurs_csfleurs_csyodas) view=csyodas; feats_type=raw_copy; max_wav_duration=70 ;;
    *) echo "Error: unknown --data ${data}" >&2; exit 1 ;;
esac
if [ "${data}" != fleurs_csfleurs ] && [ "${stage}" -le 11 ] && [ "${stop_stage}" -ge 11 ] \
    && [[ "${asr_config}" == conf/tuning/train_mms_transformer_pit*.yaml ]]; then
    echo "Error: ${asr_config} is set for --data fleurs_csfleurs;" \
        "give a copy with the values in README.md as --asr_config" >&2
    exit 1
fi
asr_tag=$(basename "${asr_config}" .yaml)_${data}_${feats_type}
inference_tag="$(basename "${inference_config}" .yaml)_asr_model_${inference_asr_model%.*}"

train_set=train_fleurscs_${view}
valid_set=valid_fleurscs_${view}
cs_test_sets="test_fleurscs_cs_read test_fleurscs_cs_xtts1 test_fleurscs_cs_xtts2 test_fleurscs_cs_mms"
test_sets="test_fleurscs_fleurs ${cs_test_sets}"

# shellcheck disable=SC2086
./asr.sh \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --ngpu "${ngpu}" \
    --nj "${nj}" \
    --inference_nj "${inference_nj}" \
    --gpu_inference true \
    --lang "lidseq_${data}" \
    --token_type word \
    --use_lm false \
    --feats_type "${feats_type}" \
    --audio_format wav \
    --feats_normalize uttmvn \
    --min_wav_duration 1.0 \
    --max_wav_duration "${max_wav_duration}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --asr_stats_dir "exp/asr_stats_${feats_type}_${view}" \
    --asr_config "${asr_config}" \
    --asr_tag "${asr_tag}" \
    --asr_args "${asr_args}" \
    --pretrained_model "${pretrained_model}" \
    --inference_config "${inference_config}" \
    --inference_tag "${inference_tag}" \
    --inference_asr_model "${inference_asr_model}" \
    ${asr_opts}

if [ "${stage}" -le 13 ] && [ "${stop_stage}" -ge 13 ]; then
    # Unordered language-set accuracy of the decoded token sequences
    decode_dir=exp/asr_${asr_tag}/${inference_tag}
    cs_refs=()
    cs_hyps=()
    for dset in ${test_sets}; do
        python3 local/score_lidseq.py --ref "data/${dset}/utt2langs" \
            --hyp "${decode_dir}/${dset}/text" --out "${decode_dir}/${dset}/lang_set_results.tsv"
        if [ "${dset}" != test_fleurscs_fleurs ]; then
            cs_refs+=("data/${dset}/utt2langs")
            cs_hyps+=("${decode_dir}/${dset}/text")
        fi
    done
    python3 local/score_lidseq.py --ref "${cs_refs[@]}" --hyp "${cs_hyps[@]}" \
        --out "${decode_dir}/test_fleurscs_cs_all/lang_set_results.tsv"
fi
