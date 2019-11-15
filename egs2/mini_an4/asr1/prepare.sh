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
    --nlsyms: non-linguistic symbol list
    --stage (int): Processes starts from the specifed stage.
    --stop-stage (int): Processes is stopped at the specifed stage.
    --oov (str): default is "<unk>"
    --token_mode (str): The tokenize level. Select either one of "bpe", "char" or "word".
    --nbpe (int):
    --lm_train_text (str): default is using the same text for asr.
    --lm_dev_text (str): default is using the same text for asr.
    --lm_eval_text (str): default is using the same text for asr.
    --srctexts (str): will be used for the training of BPE and the creation of a vocabrary list.
        default is the text of training data for asr.
EOF
)
SECONDS=0

# general configuration
stage=1
stop_stage=100

nj=50

audio_format=flac
fs=16k


token_mode=bpe
nbpe=30
# bpemode (unigram or bpe)
bpemode=unigram
bpe_input_sentence_size=100000000

# non-linguistic symbol list
nlsyms=
oov="<unk>"


lm_train_text=
lm_dev_text=
lm_eval_text=
srctexts=

# Output directory names
data_format=data_format
data_tokenized=data_tokenized


log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 4 ]; then
    log "Invalid arguments"
    log "${help_message}"
fi

. ./path.sh
. ./cmd.sh

train_set=$1
dev_set=$2
eval_sets=$3
dir=$4

for dset in "${train_set}" "${dev_set}" ${eval_sets}; do
    if [ ! -d data/"${dset}" ]; then
        log "Error: No such directory: data/${dset}"
        log "${help_message}"
        exit 1
    fi
done

if [ ! -n "${srctexts}" ]; then
    # Use the text for asr
    srctexts="data/${train_set}/text"
fi



if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Format wav.scp: data/ -> ${data_format}/"

    # ====== Recreating "wav.scp" ======
    # Kaldi-wav.scp shouldn't be used in training program,
    # which can describe the file path with unix-pipe, like "cat /some/path |",
    # but it's really undeguggable if using in training and also takes much IO cost.
    # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
    # and also it can changes the audio-format and sampling rate.

    for dset in "${train_set}" "${dev_set}" ${eval_sets}; do
        utils/copy_data_dir.sh data/${dset} ${data_format}/${dset}
        scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
            --audio-format "${audio_format}" --fs "${fs}" \
            data/${dset}/wav.scp ${data_format}/${dset}
    done
fi


bpedir=${dir}/bpe_${bpemode}${nbpe}
bpemodel=${bpedir}/model
bpedict=${bpedir}/units.txt
chardict=${dir}/char_dict/units.txt
worddict=${dir}/word_dict/units.txt



if [ "${token_mode}" = bpe ]; then
    dict="${bpedict}"

elif [ "${token_mode}" = char ]; then
    dict="${chardict}"
    log "Not yet"
    exit 1

elif [ "${token_mode}" = word ]; then
    dict="${worddict}"

    log "Not yet"
    exit 1
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if [ "${token_mode}" = bpe ]; then
        log "stage 2: Generate dict from ${srctexts} using BPE"

        mkdir -p "${bpedir}"
        cat ${srctexts} | cut -f 2- -d" "  > ${bpedir}/train.txt

        spm_train \
            --input="${bpedir}"/train.txt \
            --vocab_size="${nbpe}" \
            --model_type="${bpemode}" \
            --model_prefix="${bpemodel}" \
            --input_sentence_size="${bpe_input_sentence_size}"

        echo "<unk> 1" > "${dict}" # <unk> must be 1, 0 will be used for "blank" in CTC
        cat ${srctexts} | \
            spm_encode --model="${bpemodel}.model" --output_format=piece | \
                tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> "${dict}"

    elif [ "${token_mode}" = char ]; then
        log "stage 3: Generate character level dict from ${srctexts}"
        mkdir -p "$(dirname ${dict})"

        echo "<unk> 1" > "${dict}" # <unk> must be 1, 0 will be used for "blank" in CTC
        cat ${srctexts} | pyscripts/text/text2token.py -s 1 -n 1 -l "${nlsyms}"  \
            | cut -f 2- -d" " | tr " " "\n" | sort -u \
            | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> "${dict}"


    elif [ "${token_mode}" = word ]; then
        log "stage 3: Generate word level dict from ${srctexts}"
        mkdir -p "$(dirname ${dict})"

        log "Not yet"
        exit 1


    else
        log "Error: not supported --token_mode ${token_mode}"
        exit 1
    fi

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Create tokens from text"

    for dset in "${train_set}" "${dev_set}" ${eval_sets}; do
        # 1. Copy datadir
        utils/copy_data_dir.sh ${data_format}/${dset} ${data_tokenized}/${dset}
        # These are extended files from Kaldi-datadir
        for f in utt2num_samples; do
            [ -e "${data_format}/${dset}/${f}" ] && cp "${data_format}/${dset}/${f}" "${data_tokenized}/${dset}/${f}"
        done

        scripts/asr/prepare_token.sh \
            --mode "${token_mode}" --bpemodel "${bpemodel}.model" \
                ${data_format}/${dset}/text ${dict} ${data_tokenized}/${dev_set}
    done


    if [ -n "${lm_train_text}" ]; then
        scripts/asr/prepare_token.sh \
            --mode "${token_mode}" --bpemodel "${bpemodel}.model" \
                "${lm_train_text}" "${dict}" "${data_tokenized}"/lm_train
    fi
    if [ -n "${lm_dev_text}" ]; then
        scripts/asr/prepare_token.sh \
            --mode "${token_mode}" --bpemodel "${bpemodel}.model" \
                "${lm_dev_text}" "${dict}" "${data_tokenized}"/lm_dev
    fi
    if [ -n "${lm_eval_text}" ]; then
        scripts/asr/prepare_token.sh \
            --mode "${token_mode}" --bpemodel "${bpemodel}.model" \
                "${lm_eval_text}" "${dict}" "${data_tokenized}"/lm_eval
    fi
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
