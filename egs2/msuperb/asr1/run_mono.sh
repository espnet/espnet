#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Process Pipeline
stage=1
stop_stage=13
nj=32
inference_nj=32
gpu_inference=false

# Model/Inference Configs
asr_config=conf/tuning/train_asr_fbank_single.yaml
inference_config=conf/decode_asr.yaml

./utils/parse_options.sh || exit 1

for duration in 10min 1h; do
    for single_lang in eng1 eng2 eng3 fra1 fra2 deu1 deu2 rus swa swe jpn cmn xty ; do
        echo "processing ${single_lang} ${duration}"
        train_set=train_${duration}_${single_lang}
        train_dev=dev_10min_${single_lang}
        test_set="${train_dev} test_10min_${single_lang}"
        asr_tag="$(basename "${asr_config}" .yaml)_${single_lang}_${duration}"

        if [ "${single_lang}" == "cmn" ] || [ "${single_lang}" == "jpn" ]; then
            token_type=word
        else
            token_type=char
        fi

	local_data_opts="--duration ${duration} --lid false --multilingual false "
	local_data_opts+="--single_lang ${single_lang} --nlsyms_txt ${nlsyms_txt}"

        ./asr.sh \
            --ngpu 1 \
	    --stage ${stage} \
            --stop_stage ${stop_stage} \
	    --nj ${nj} \
	    --inference_nj ${inference_nj} \
	    --gpu_inference ${gpu_inference} \
            --lang ${single_lang} \
            --inference_asr_model "valid.loss.ave.pth" \
            --local_data_opts "${local_data_opts}" \
            --use_lm false \
            --token_type ${token_type} \
            --feats_type raw \
            --feats_normalize utterance_mvn \
            --asr_config "${asr_config}" \
            --inference_config "${inference_config}" \
            --train_set "${train_set}" \
            --valid_set "${train_dev}" \
            --test_sets "${test_set}" \
            --asr_tag "${asr_tag}" \
            --asr_stats_dir exp/asr_stats_${lang}_${duration} \
            --local_score_opts "false false monolingual"
    done
done
