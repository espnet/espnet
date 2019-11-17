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
help_message=$(cat << EOF
$0 <train_set_name> <dev_set_name> <eval_set_names> <output-dir>

Options:
    --nj (int): The number of parallel jobs
    --stage (int): Processes starts from the specifed stage.
    --stop-stage (int): Processes is stopped at the specifed stage.
    --nlsyms: non-linguistic symbol list
    --oov (str): Out of vocabrary symbol. The default is "<unk>"
    --token_mode (str): The tokenize level. Select either one of "bpe", "char" or "word". The default is "bpe"
    --nbpe (int):
EOF
)
SECONDS=0


## general configuration
stage=1
stop_stage=100

nj=50
decode_nj=50
# number of gpus ("0" uses cpu, otherwise use gpu)
ngpu=1
gpu_decode=false


## The options for prepare.sh
audio_format=flac
fs=16k

# token_mode=char
token_mode=bpe

nbpe=30
# bpemode (unigram or bpe)
bpemode=unigram
bpe_input_sentence_size=100000000
oov="<unk>"



## The options for training
exp=exp

# Add suffix to the result dir for lm training
lm_tag=
lm_config=
lm_preprocess_config=
use_word_lm=false
word_vocab_size=10000

# Add suffix to the result dir for asr training
asr_tag=
asr_config=
asr_preprocess_config=


. utils/parse_options.sh
. ./path.sh
. ./cmd.sh

# [Task depented] Set the datadir name created by local/data.sh
train_set=train_nodev
dev_set=train_dev
eval_sets="test "
# non-linguistic symbol list if existing
nlsyms=


# srctexts, which will be used for the training of BPE and the creation of a vocabrary list,
# can be set as multiple texts.
srctexts="data/${train_set}/text "

lm_train_text="data/${train_set}/text "
lm_dev_text="data/${dev_set}/text "
lm_eval_text="data/${eval_sets%% *}/text "

data_format=data_format
data_tokenized="data_tokenized_${token_mode}"
if ${use_word_lm}; then
    data_lm=data_${token_mode}_lm
else
    data_lm=data_word_lm
fi

dictdir=data/local
bpedir="${dictdir}/bpe_${bpemode}${nbpe}"
bpemodel="${bpedir}"/model
bpedict="${bpedir}"/units.txt
chardict="${dictdir}"/char_dict/units.txt
worddict="${dictdir}"/word_dict/units.txt


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparation: data/${train_set}, data/${dev_set}, etc."

    # TODO(kamo): Change Makefile to install sph2pipe
    local/data.sh --sph2pipe "${KALDI_ROOT}/tools/sph2pipe_v2.5/sph2pipe"
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Format wav.scp: data/ -> ${data_format}/"

    # ====== Recreating "wav.scp" ======
    # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
    # shouldn't be used in training process, although it's supported.
    # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
    # and also it can also change the audio-format and sampling rate.
    # If nothing is need, then format_wav_scp.sh does nothing:
    # i.e. the input file format and rate is same as the output.

    for dset in "${train_set}" "${dev_set}" ${eval_sets}; do
        utils/copy_data_dir.sh data/"${dset}" "${data_format}/${dset}"
        scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
            --audio-format "${audio_format}" --fs "${fs}" \
            "data/${dset}/wav.scp" "${data_format}/${dset}"
    done
fi


if [ "${token_mode}" = bpe ]; then
    dict="${bpedict}"
elif [ "${token_mode}" = char ]; then
    dict="${chardict}"
else
    log "Error: not supported --token_mode ${token_mode}"
    exit 1
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    if [ "${token_mode}" = bpe ]; then
        log "stage 3: Generate dict from ${srctexts} using BPE"

        mkdir -p "${bpedir}"
        cat ${srctexts} | cut -f 2- -d" "  > "${bpedir}"/train.txt

        spm_train \
            --input="${bpedir}"/train.txt \
            --vocab_size="${nbpe}" \
            --model_type="${bpemode}" \
            --model_prefix="${bpemodel}" \
            --input_sentence_size="${bpe_input_sentence_size}"

        echo "<unk> 1" > "${dict}" # <unk> must be 1, 0 will be used for "blank" in CTC
        cat ${srctexts} | \
            spm_encode --model="${bpemodel}.model" --output_format=piece | \
                tr ' ' '\n' | sort -u | awk '{print $0 " " NR+1}' >> "${dict}"

    elif [ "${token_mode}" = char ]; then
        log "stage 3: Generate character level dict from ${srctexts}"
        mkdir -p "$(dirname ${dict})"


        echo "<unk> 1" > "${dict}" # <unk> must be 1, 0 will be used for "blank" in CTC
        if [ -n "${nlsyms}" ]; then
            cat ${srctexts} | pyscripts/text/text2token.py -s 1 -n 1 -l "${nlsyms}"  \
                | cut -f 2- -d" " | tr " " "\n" | sort -u \
                | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> "${dict}"
        else
            cat ${srctexts} | pyscripts/text/text2token.py -s 1 -n 1 \
                | cut -f 2- -d" " | tr " " "\n" | sort -u \
                | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> "${dict}"
        fi
    else
        log "Error: not supported --token_mode ${token_mode}"
        exit 1
    fi

    if ${use_word_lm}; then
        log "Generate word level dict from ${srctexts}"
        mkdir -p "$(dirname ${dict})"

        cat ${srctexts} | pyscripts/text/text2vocabulary.py -s ${word_vocab_size} -o ${dict}
    fi

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Create tokens from text: dict=${dict}"

    for dset in "${train_set}" "${dev_set}" ${eval_sets}; do
        # 1. Copy datadir
        utils/copy_data_dir.sh "${data_format}/${dset}" "${data_tokenized}/${dset}"
        # Copying extended files from Kaldi-datadir
        [ -e "${data_format}/${dset}/utt2num_samples" ] && \
            cp "${data_format}/${dset}/utt2num_samples" "${data_tokenized}/${dset}/utt2num_samples"

        scripts/asr/prepare_token.sh \
            --mode "${token_mode}" --bpemodel "${bpemodel}.model" \
                "${data_format}/${dset}/text" "${dict}" "${data_tokenized}/${dset}"
    done

    if ${use_word_lm}; then
        _lm_token_mode="word"
    else
        _lm_token_mode="${token_mode}"
    fi

    scripts/asr/prepare_token.sh \
        --mode "${_lm_token_mode}" --bpemodel "${bpemodel}.model" \
            "${lm_train_text}" "${dict}" "${data_lm}"/train
    scripts/asr/prepare_token.sh \
        --mode "${_lm_token_mode}" --bpemodel "${bpemodel}.model" \
            "${lm_dev_text}" "${dict}" "${data_lm}"/dev
    scripts/asr/prepare_token.sh \
        --mode "${_lm_token_mode}" --bpemodel "${bpemodel}.model" \
            "${lm_eval_text}" "${dict}" "${data_lm}"/eval
fi

# ========================== Data preparation is done here. ==========================


lm_exp="${exp}/${token_mode}lm_train${lm_tag}"
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "stage 5: LM Training: train_set=${data_lm}/train, dev_set=${data_lm}/dev"

    _opts=
    if [ -n "${lm_config}" ]; then
        # To generate the config file: e.g.
        #   % python -m espnet2.bin.train lm --print_config --optimizer adam --lm transformer
        _opts+="--config ${lm_config} "
    fi
    if [ -n "${lm_preprocess_config}" ]; then
        _opts+="--train_preprosess input=${lm_preprocess_config} "
        _opts+="--eval_preprosess input=${lm_preprocess_config} "
    fi

    log "LM training started... log: ${lm_exp}/train.log"
    ${cuda_cmd} --gpu "${ngpu}" "${lm_exp}"/train.log \
        python3 -m espnet2.bin.lm_train \
            --ngpu "${ngpu}" \
            --token_list "${data_lm}/train/tokens.txt" \
            --train_data_path_and_name_and_type "${data_lm}/train/token_int,input,text_int" \
             --eval_data_path_and_name_and_type "${data_lm}/dev/token_int,input,text_int" \
            --train_shape_file "${data_lm}/train/token_shape" \
             --eval_shape_file "${data_lm}/dev/token_shape" \
            --output_dir "${lm_exp}"

fi


asr_exp="${exp}/asr_train${asr_tag}"
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    asr_train_dir="${data_tokenized}/${train_set}"
    asr_dev_dir="${data_tokenized}/${dev_set}"
    log "stage 6: ASR Training: train_set=${asr_train_dir}, dev_set=${asr_dev_dir}"

    _opts=
    if [ -n "${asr_config}" ]; then
        # To generate the config file: e.g.
        #   % python -m espnet2.bin.train asr --print_config --optimizer adam --encoder_decoder transformer
        _opts+="--config ${asr_config} "
    fi
    if [ -n "${asr_preprocess_config}" ]; then
        # syntax: --train_preprosess {key}={yaml file or yaml string}
        _opts+="--train_preprosess input=${asr_preprocess_config} "
         _opts+="--eval_preprosess input=${asr_preprocess_config} "
    fi

    log "ASR training started... log: ${asr_exp}/train.log"
    ${cuda_cmd} --gpu "${ngpu}" "${asr_exp}"/train.log \
        python3 -m espnet2.bin.asr_train \
            ${_opts} \
            --ngpu "${ngpu}" \
            --token_list "${asr_train_dir}/tokens.txt" \
            --train_data_path_and_name_and_type "${asr_train_dir}/wav.scp,input,sound" \
            --train_data_path_and_name_and_type "${asr_train_dir}/token_int,output,text_int" \
             --eval_data_path_and_name_and_type "${asr_dev_dir}/wav.scp,input,sound" \
             --eval_data_path_and_name_and_type "${asr_dev_dir}/token_int,output,text_int" \
            --train_shape_file "${asr_train_dir}/utt2num_samples" \
            --train_shape_file "${asr_train_dir}/token_shape" \
             --eval_shape_file "${asr_dev_dir}/utt2num_samples" \
             --eval_shape_file "${asr_dev_dir}/token_shape" \
            --output_dir "${asr_exp}"

fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "stage 7: Decoding"

    if ${gpu_decode}; then
        _cmd=${cuda_cmd}
        _ngpu=1
    else
        _cmd=${decode_cmd}
        _ngpu=0
    fi

    _opts=
    if [ -n "${asr_preprocess_config}" ]; then
        _opts+="--preprosess input=${asr_preprocess_config} "
    fi
    if [ -f "${asr_exp}"/acc.best.pt ]; then
        _asr_model_file="${asr_exp}"/acc.best.pt
    else
        _asr_model_file="${asr_exp}"/loss.best.pt
    fi

    for dset in ${eval_sets}; do
        key_file=data/${dset}/wav.scp
        logdir="${asr_exp}/decode_${dset}/logdir"
        split_scps=""
        for n in $(seq ${nj}); do
            split_scps="${split_scps} ${logdir}/keys.${n}.scp"
        done
        utils/split_scp.pl "${key_file}" ${split_scps}

        ${_cmd} --gpu "${_ngpu}" JOB=1:"${decode_nj}" \
            python3 -m espnet2.bin.asr_recog \
                ${_opts} \
                --ngpu "${_ngpu}" \
                --output_dir "${logdir}"/output.JOB \
                --data_path_and_name_and_type "data/${dset}/wav.scp,input,sound" \
                --key_file "${logdir}"/keys.JOB.scp \
                --asr_train_config "${asr_exp}"/config.yaml \
                --asr_model_file "${_asr_model_file}" \
                --lm_train_config "${lm_exp}"/config.yaml \
                --lm_file "${lm_exp}"/loss.best.pt

            for f in token token_int text score; do
                for i in $(seq "${decode_nj}"); do
                    cat "${logdir}/output.${i}/${f}"
                done | LC_ALL=C sort -k1 >"${asr_exp}/decode_${dset}/${f}"
            done
    done
    scripts/asr/show_result.sh ${exp}

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
