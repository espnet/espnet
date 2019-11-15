#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $@"
}
SECONDS=0


## general configuration
stage=1
stop_stage=100

# number of gpus ("0" uses cpu, otherwise use gpu)
nj=50


## The options for prepare.sh
audio_format=flac
fs=16k

# token_mode=char
# token_mode=word
token_mode=bpe

nbpe=30
# bpemode (unigram or bpe)
bpemode=unigram
bpe_input_sentence_size=100000000
# non-linguistic symbol list
nlsyms=
oov="<unk>"



## The options for training
exp=exp

# Add suffix to the result dir for lm training
ngpu=0
lm_tag=
lm_config=
lm_preprocess_config=

# Add suffix to the result dir for asr training
asr_tag=
asr_config=
asr_preprocess_config=


. utils/parse_options.sh
. ./path.sh
. ./cmd.sh


train_set=train_nodev
dev_set=train_dev
eval_sets=test

# srctexts, which will be used for the training of BPE and the creation of a vocabrary list,
# can be set as multiple texts.
srctexts="data/${train_set}/text "

# If anothet
lm_train_text="data/${train_set}/text"
lm_dev_text="data/${dev_set}/text"
lm_eval_text="data/${eval_sets%% *}/text"


data_format=data_format
data_tokenized=data_tokenized_${token_mode}


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparation: data/${train_set}, data/${dev_set}, etc."

    # TODO(kamo): Change Makefile to install sph2pipe
    local/data.sh --sph2pipe "${KALDI_ROOT}/tools/sph2pipe_v2.5/sph2pipe"
fi



if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Prepare.sh"
    ./prepare.sh  \
        --stage 1 \
        --stop-stage 1000000 \
        --nj "${nj}" \
        --audio_format "${audio_format}" \
        --fs "${fs}" \
        --token_mode "${token_mode}" \
        --nbpe "${nbpe}" \
        --bpemode "${bpemode}" \
        --bpe_input_sentence_size "${bpe_input_sentence_size}" \
        --nlsyms "${nlsyms}" \
        --oov "${oov}" \
        --lm_train_text "${lm_train_text}" \
        --lm_dev_text "${lm_dev_text}" \
        --lm_eval_text "${lm_eval_text}" \
        --data_format "${data_format}" \
        --data_tokenized "${data_tokenized}" \
        ${train_set} ${dev_set} "${eval_sets}" data/local
fi
# ========================== Data preparation is done here. ==========================


lm_exp="${exp}/${token_mode}lm_train${lm_tag}"
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: LM Preparation"

    if [ -n "${lm_train_text}" ]; then
        lm_train_dir="${data_tokenized}/lm_train"
    else
        lm_train_dir="${data_tokenized}/${train_set}"
    fi
    if [ -n "${lm_dev_text}" ]; then
        lm_dev_dir="${data_tokenized}/lm_dev"
    else
        lm_dev_dir="${data_tokenized}/${dev_set}"
    fi
    if [ -n "${lm_eval_text}" ]; then
        lm_eval_dir="${data_tokenized}/lm_eval"
    else
        lm_eval_dir="${data_tokenized}/${eval_set}"
    fi

    _opts=
    if [ -n "${lm_config}" ]; then
        # To generate the config file: e.g.
        #   % python -m espnet2.bin.train lm --print_config --optimizer adam --lm transformer
        # Note that the configuration is changed according to the specified class type. i.e. adam, transformer in this case.
        _opts+="--config ${lm_config} "
    fi
    if [ -n "${lm_preprocess_config}" ]; then
        _opts+="--train_preprosess input=${lm_preprocess_config} "
        _opts+="--eval_preprosess input=${lm_preprocess_config} "
    fi


    log "LM training started... log: ${lm_exp}/train.log"
    ${cuda_cmd} --gpu "${ngpu}" "${lm_exp}"/train.log \
        python -m espnet2.bin.train lm \
            --ngpu "${ngpu}" \
            --token_list "${lm_train_dir}/tokens.txt" \
            --train_data_conf input.path="${lm_train_dir}/token_int" \
            --train_data_conf input.type=text_int \
            --train_batch_files "[${lm_train_dir}/token_shape]" \
            --eval_data_conf input.path="${lm_dev_dir}/token" \
            --eval_data_conf input.type=text_int \
            --eval_batch_files "[${lm_dev_dir}/token_shape]" \
            --output_dir "${lm_exp}"

fi


asr_exp="${exp}/asr_train${asr_tag}"
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "stage 5: Network Training"

    asr_train_dir="${data_tokenized}/${train_set}"
    asr_dev_dir="${data_tokenized}/${dev_set}"

    _opts=
    if [ -n "${asr_config}" ]; then
        # To generate the config file: e.g.
        #   % python -m espnet2.bin.train asr --print_config --optimizer adam --encoder_decoder transformer
        # Note that the configuration is changed according to the specified class type. i.e. adam, transformer in this case.
        _opts+="--config ${asr_config} "
    fi
    if [ -n "${asr_preprocess_config}" ]; then
        _opts+="--train_preprosess input=${asr_preprocess_config} "
        _opts+="--eval_preprosess input=${asr_preprocess_config} "
    fi

    # token_shape: <uttid> <olength>,<odim>: e.g. uttidA 12,27
    odim="$(<${asr_train_dir}/token_shape awk 'NR==1 {split($2,sp,",");print(sp[2]);}')"
    if ! python3 -c "assert isinstance(${odim}, int)" &> /dev/null; then
        log "Error: '${odim}' is not an integer number."
        exit 1
    fi

    log "ASR training started... log: ${asr_exp}/train.log"
    ${cmd} --gpu "${ngpu}" "${asr_exp}/train.log" \
        python3 -m espnet2.bin.train asr \
            ${_opts} \
            --ngpu "${ngpu}" \
            --odim "${odim}" \
            --fs "${fs}" \
            --token_list "${asr_train_dir}/tokens.txt" \
            --train_data_conf input.path="${asr_train_dir}/wav.scp" \
            --train_data_conf input.type=sound \
            --train_data_conf output.path="${asr_train_dir}/token_int" \
            --train_data_conf output.type=text_int \
            --train_batch_files "[${asr_train_dir}/utt2num_samples, ${asr_train_dir}/token_shape]" \
            --eval_data_conf input.path="${asr_dev_dir}/wav.scp" \
            --eval_data_conf input.type=sound \
            --eval_data_conf output.path="${asr_dev_dir}/token_int" \
            --eval_data_conf output.type=text_int \
            --eval_batch_files "[${asr_dev_dir}/utt2num_samples, ${asr_dev_dir}/token_shape]" \
            --output_dir "${asr_exp}"

fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "stage 5: Decoding"
    log "Not yet"
    exit 1

    if ${gpu_decode}; then
        _cmd=${cuda_cmd}
        _ngpu=1
    else
        _cmd=${decode_cmd}
        _ngpu=0
    fi

    for dset in ${eval_sets}; do
        scripts/asr/decode.sh --cmd "${_cmd} --gpu ${ngpu}" --nj "${decode_nj}" --ngpu "${ngpu}" \
            ${exp}/results data/${dset} ${exp}/decode_${dset}
    done

    scripts/asr/show_result.sh ${exp}

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
