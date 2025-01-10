#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=100000
flac2wav=false
nj=32


log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${MYST}" ]; then
    log "Fill the value of 'MYST' in db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -d "${MYST}" ]; then
        log "stage 1: Please download data from https://catalog.ldc.upenn.edu/LDC2021S05 and save to ${MYST}"
        exit 1
    else
        log "stage 1: ${MYST} already exists. Skipping data downloading."
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if "${flac2wav}"; then
        log "stage 2: Convert flac to wav"
        original_dir="${MYST}/data"
        logdir="${MYST}/myst_child_conv_speech/log"
        mkdir -p $logdir
        cmd=${train_cmd}
        ${cmd} "JOB=1:1" "${logdir}/flac_to_wav.JOB.log" \
            python local/flac_to_wav.py \
                --multiprocessing \
                --njobs ${nj} \
                --myst_dir ${original_dir}
    else
        log "flac2wav is false. Skipping conversion."
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Data preparation"
    original_dir="${MYST}/data"
    data_dir="./data"

    if "${flac2wav}"; then
        python local/prepare_data.py --original-dir $original_dir --data-dir $data_dir --is-wav
    else
        python local/prepare_data.py --original-dir $original_dir --data-dir $data_dir
    fi

    partitions="train dev test"
    for dset in $partitions; do
        data_partition=$data_dir/$dset
        text_file=$data_partition/text
        utt2spk_file=$data_partition/utt2spk
        wav_scp_file=$data_partition/wav.scp
        spk2utt_file=$data_partition/spk2utt
        for f in $text_file $utt2spk_file $wav_scp_file; do
            sort "$f" -o "$f"
        done
        utils/utt2spk_to_spk2utt.pl "$utt2spk_file" > "$spk2utt_file"
        iconv -f utf-8 -t ascii//TRANSLIT "$text_file" > "${text_file}.ascii"
        mv "${text_file}.ascii" "$text_file"
        utils/validate_data_dir.sh --no-feats "$data_partition"
    done
    log "stage 3: Data preparation completed."
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Transcription and scoring"

    WER_Threshold=0.5
    filter_n_words=3
    model_name="openai/whisper-large-v2"
    for x in train dev test; do

        datalist=conf/file_list/${x}.lst

        if [ -f $datalist ]; then
            log "Filtering files in $x set"

            utils/subset_data_dir.sh --utt-list "${datalist}" "${data_dir}/${x}" "${data_dir}/${x}_filter"

        else
            log "No file list found for $x set"

            if [ "${x}" == "test" ]; then
                remove_long_duration=300
            else
                remove_long_duration=30
            fi
            wav_scp="${data_dir}/${x}/wav.scp"
            trn_scp="${data_dir}/${x}/text"
            filtered_list="${data_dir}/${x}/${x}.lst"

            # Filter utterances by WER < 50%
            log "Filtering utterances by WER for ${x}..."
            python local/whisper_filter.py \
                --wav_scp $wav_scp \
                --trn_scp $trn_scp\
                --model $model_name \
                --wer_threshold $WER_Threshold \
                --remove_n_words $filter_n_words \
                --remove_long_dur $remove_long_duration \
                --saved_utt_list $filtered_list
            # Subset filtered data
            log "Subsetting filtered data for ${x}..."
            utils/subset_data_dir.sh --utt-list "${filtered_list}" "${data_dir}/${x}" "${data_dir}/${x}_filter"

        fi
    done
    log "Stage 4: Filtering completed."
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "stage 5: Text normalization"
    for x in train_filter dev_filter test_filter; do
        input_text=data/$x/text
        output_text=data/$x/text_normalized
        python local/text_preprocess.py $input_text $output_text
        mv $output_text $input_text
    done
    log "stage 5: Text normalization completed."
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
