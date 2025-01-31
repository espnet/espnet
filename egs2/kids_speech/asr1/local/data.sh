#!/usr/bin/env bash

# set -e
# set -u
# set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=100000

flac2wav=true
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
    log "Fill the value of 'MYST' of db.sh"
    exit 2
fi

if [ -z "${OGI_KIDS}" ]; then
    log "Fill the value of 'OGI_KIDS' in db.sh"
    exit 1
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -d "${OGI_KIDS}" ]; then
        log "stage 1: Please download data from https://catalog.ldc.upenn.edu/LDC2007S18 and save to ${OGI_KIDS}"
        exit 1

    elif [ ! -d "${MYST}/myst_child_conv_speech" ]; then
	    log "stage 1: Please download data from https://catalog.ldc.upenn.edu/LDC2021S05 and save to ${MYST}"
        exit 1

    else
        log "stage 1: ${OGI_KIDS} already exists. Skipping data downloading."
    fi
fi

# this stage is for MyST
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if "${flac2wav}"; then
        log "stage 2: Convert flac to wav"
        original_dir="${MYST}/myst_child_conv_speech/data"
        logdir="${MYST}/myst_child_conv_speech/log"
        mkdir -p $logdir
        cmd=${train_cmd}
        ${cmd} "JOB=1:1" "${logdir}/flac_to_wav.JOB.log" \
            python local/flac_to_wav.py \
                --multiprocessing \
                --njobs ${nj} \
                --myst_dir ${original_dir}
    else
        log "flac2wav is false. Skip convertion."
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Data preparation"

    # prepare data for MyST
    myst_dir="${MYST}/myst_child_conv_speech/data"
    myst_data_dir="./data_myst"
    
    if "${flac2wav}"; then
        python local/prepare_data.py --original-dir $myst_dir --data-dir $myst_data_dir --is-wav
    else
        python local/prepare_data.py --original-dir $myst_dir --data-dir $myst_data_dir
    fi
    log "MyST data prepared."

    # 2. Prefix the speaker ID with 'myst_'; TODO: simplify this part
    for split in train dev test; do
      if [ -f "$myst_data_dir/$split/utt2spk" ]; then
        # Make a backup
        mv "$myst_data_dir/$split/utt2spk" "$myst_data_dir/$split/utt2spk.org"
        # Transform the second column to prefix it with 'myst_'
        awk '{print $1, "myst_"$2}' "$myst_data_dir/$split/utt2spk.org" > "$myst_data_dir/$split/utt2spk"

        # (Re-generate spk2utt to match the new speaker IDs)
        utils/utt2spk_to_spk2utt.pl "$myst_data_dir/$split/utt2spk" \
          > "$myst_data_dir/$split/spk2utt"
      fi
    done

    # prepare data for ogi scripted
    ogi_dir="${OGI_KIDS}"
    ogi_scripted_data_dir="./data_ogi_scripted"
    # local/ogi_scripted_prepare.sh $ogi_dir $ogi_scripted_data_dir
    log "OGI kids scripted data prepared."

    # # prepare all spon data without considering segment length
    ogi_spon_data_dir="./data_ogi_spon"
    local/ogi_spon_all_data_prepare.sh $ogi_dir/ $ogi_spon_data_dir/

    # prepare data for ogi spontaneous
    log "stage 3: Data preparation completed."
fi

# this stage is for OGI-scripted
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: CTC Segment"

    #lists dir is used to ensure train, dev, test splits are done based on speakers from scripted ogi
    lists_dir="conf/file_list"

    python local/ctc_segment.py --input $ogi_spon_data_dir/spont_all --lists $lists_dir --output $ogi_spon_data_dir

    python create_utt2spk.py

    log "Stage 4: Finished ctc segmentation"
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Merging datasets"

    output_dir="./data"

    # Create the output directory
    mkdir -p $output_dir/train $output_dir/dev $output_dir/test

    # Initialize the Kaldi format files
    for dir in train dev test; do
    : > "$output_dir/$dir/wav.scp"
    : > "$output_dir/$dir/text"
    : > "$output_dir/$dir/utt2spk"
    done

    mv -r "$ogi_spon_data_dir/wav" "$output_dir/wav"

    # List of datasets
    datasets=("$myst_data_dir" "$ogi_scripted_data_dir" "$ogi_spon_data_dir")

    # Merge wav.scp
    for dataset in "${datasets[@]}"; do
        cat "$dataset/train/wav.scp" >> data/train/wav.scp
        cat "$dataset/dev/wav.scp" >> data/dev/wav.scp
        cat "$dataset/test/wav.scp" >> data/test/wav.scp
    done

    # Merge text
    for dataset in "${datasets[@]}"; do
        cat "$dataset/train/text" >> data/train/text
        cat "$dataset/dev/text" >> data/dev/text
        cat "$dataset/test/text" >> data/test/text
    done

    # Merge utt2spk
    for dataset in "${datasets[@]}"; do
        cat "$dataset/train/utt2spk" >> data/train/utt2spk
        cat "$dataset/dev/utt2spk" >> data/dev/utt2spk
        cat "$dataset/test/utt2spk" >> data/test/utt2spk
    done

    # Sort and clean
    for split in train dev test; do
        for f in text wav.scp utt2spk; do
            sort data/${split}/${f} -o data/${split}/${f}
        done

        # # sort text, wav.scp by default
        # for f in text wav.scp; do
        #     sort data/${split}/${f} -o data/${split}/${f}
        # done
        # # sort utt2spk by speaker ID
        # sort -k2,2 data/${split}/utt2spk -o data/${split}/utt2spk

        utils/utt2spk_to_spk2utt.pl "data/$split/utt2spk" > "data/$split/spk2utt"

        dos2unix "data/${split}/text"

        # # Remove utf-8 whitespaces
        iconv -f utf-8 -t ascii//TRANSLIT "data/${split}/text" > "data/${split}/text.ascii"
        mv "data/${split}/text.ascii" "data/${split}/text"

        # Validate data
        utils/validate_data_dir.sh --no-feats "data/${split}"
    done

    log "Stage 5: Merging datasets completed."
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
