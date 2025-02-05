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
sph2wav=true
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

if [ -z "${CMU_KIDS}" ]; then
    log "Fill the value of 'CMU_KIDS' in db.sh"
    exit 1
fi

myst_dir="${MYST}/myst_child_conv_speech/data"
ogi_dir="${OGI_KIDS}"
cmu_dir="${CMU_KIDS}"

output_dir="./data"
myst_data_dir="./data_myst"
ogi_scripted_data_dir="./data_ogi_scripted"
ogi_spon_data_dir="./data_ogi_spon"
cmu_data_dir="${output_dir}/data_cmu"

# myst file lists is for filtering, the other three are for splits
myst_lists_dir="local/myst_file_list"
ogi_scripted_lists_dir="local/ogi_scripted_file_list"
ogi_spon_lists_dir="local/ogi_spon_file_list"
cmu_lists_dir="local/cmu_file_list"

mkdir -p $output_dir


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -d "${OGI_KIDS}" ]; then
        log "stage 1: Please download data from https://catalog.ldc.upenn.edu/LDC2007S18 and save to ${OGI_KIDS}"
        exit 1
    elif [ ! -d "${MYST}/myst_child_conv_speech" ]; then
	    log "stage 1: Please download data from https://catalog.ldc.upenn.edu/LDC2021S05 and save to ${MYST}"
        exit 1
    elif [ ! -d "${CMU_KIDS}" ]; then
        log "stage 1: Please download data from https://catalog.ldc.upenn.edu/LDC97S63 and save to ${CMU_KIDS}"
        exit 1
    else
        log "stage 1: ${OGI_KIDS}, ${MYST}, and ${CMU_KIDS} already exists. Skipping data downloading."
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
    if "${flac2wav}"; then
        python local/prepare_data.py --original-dir $myst_dir --data-dir $myst_data_dir --is-wav
    else
        python local/prepare_data.py --original-dir $myst_dir --data-dir $myst_data_dir
    fi
    
    # TODO: simplify this part
    for split in train dev test; do
      if [ -f "$myst_data_dir/$split/utt2spk" ]; then
        mv "$myst_data_dir/$split/utt2spk" "$myst_data_dir/$split/utt2spk.org"
        awk '{print $1, "myst_"$2}' "$myst_data_dir/$split/utt2spk.org" > "$myst_data_dir/$split/utt2spk"

        utils/utt2spk_to_spk2utt.pl "$myst_data_dir/$split/utt2spk" \
          > "$myst_data_dir/$split/spk2utt"
      fi
    done
    log "MyST data prepared."

    # prepare data for ogi scripted
    local/ogi_scripted_prepare.sh $ogi_dir $ogi_scripted_data_dir $ogi_scripted_lists_dir
    log "OGI kids scripted data prepared."

    # prepare data for ogi spontaneous
    #local/ogi_spon_all_data_prepare.sh $ogi_dir/ $ogi_spon_data_dir/
    log "OGI kids spontaneous data prepared."

    # prepare data for cmu kids
    local/cmu_kids_data_prepare.sh $cmu_dir $cmu_data_dir $cmu_lists_dir
    log "CMU kids data prepared."
    log "stage 3: Data preparation completed."
fi

# Filter data in MyST, >50% Whisper WER, shorter than 0.5 sec or shorter than 3 words
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Filtering MyST"

    for x in train dev test; do

        datalist=${myst_lists_dir}/${x}.list
        log "Filtering files in $x set"
        utils/subset_data_dir.sh --utt-list "${datalist}" "${myst_data_dir}/${x}" "${myst_data_dir}/${x}_filter"
        
        mv ${myst_data_dir}/${x} ${myst_data_dir}/${x}_org
        mv ${myst_data_dir}/${x}_filter ${myst_data_dir}/${x}
    done

    # TODO: add whisper_filter.py here?

    log "Stage 4: MyST filtered."
fi

# this stage is for OGI-scripted
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: CTC Segment"

    #python local/ctc_segment.py --input $ogi_spon_data_dir/spont_all --lists $ogi_spon_lists_dir --output $ogi_spon_data_dir
    python local/create_utt2spk.py

    log "Stage 5: Finished ctc segmentation"
fi

# sph to wav conversion for cmu kids
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Wav conversion"

    if $sph2wav; then
        log "Converting .sph to .wav"

        for x in train dev test; do
            mkdir -p ${cmu_data_dir}/${x}/wav
            true > ${cmu_data_dir}/${x}/wav.scp.new

            while IFS=' ' read -r uttID wavCmd; do
                sphFile=$(echo $wavCmd | sed -e 's/.*sph2pipe -f wav -p -c 1 //; s/|$//')
                wavFile="${cmu_data_dir}/${x}/wav/${uttID}.wav"
                python local/sph2wav.py --input $sphFile --output $wavFile
                echo "$uttID $wavFile" >> ${cmu_data_dir}/${x}/wav.scp.new
            done < ${cmu_data_dir}/${x}/wav.scp
            rm ${cmu_data_dir}/${x}/wav.scp
            mv ${cmu_data_dir}/${x}/wav.scp.new ${cmu_data_dir}/${x}/wav.scp

        done
        log "Stage 6: Finished .sph to .wav conversion"
    else
        log "Stage 6: .sph to .wav conversion skipped."
    fi
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Merging datasets"

    # Create the output directory
    mkdir -p $output_dir/train $output_dir/dev $output_dir/test

    # Initialize the Kaldi format files
    for dir in train dev test; do
        : > "$output_dir/$dir/wav.scp"
        : > "$output_dir/$dir/text"
        : > "$output_dir/$dir/utt2spk"
        # mv "$cmu_data_dir/$dir/wav" "$output_dir/$dir/wav" # TODO: but in the wav.scp, the path is this
    done

    mv "$ogi_spon_data_dir/wav" "$output_dir/wav"

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

    # merge cmu_kids; TODO: make the code in same style
    for split in train dev test; do
        for f in text wav.scp utt2spk; do
            cat "${cmu_data_dir}/${split}/${f}"
        done
    done

    # Sort and clean
    for split in train dev test; do
        for f in text wav.scp utt2spk; do
            sort data/${split}/${f} -o data/${split}/${f}
        done

        utils/utt2spk_to_spk2utt.pl "data/$split/utt2spk" > "data/$split/spk2utt"

        dos2unix "data/${split}/text"

        # TODO: change these hard-code data directory to output_dir
        # # Remove utf-8 whitespaces
        iconv -f utf-8 -t ascii//TRANSLIT "data/${split}/text" > "data/${split}/text.ascii"
        mv "data/${split}/text.ascii" "data/${split}/text"

        # Validate data
        utils/validate_data_dir.sh --no-feats "data/${split}"
    done

    log "Stage 7: Merging datasets completed."
fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    log "Stage 8: Creating test sets"
    # create separate test sets; TODO: combine these code with the code above
    datasets=("$myst_data_dir" "$ogi_scripted_data_dir" "$ogi_spon_data_dir")
    for dataset in "${datasets[@]}"; do
        for split in dev test; do
            for f in text wav.scp utt2spk; do
                sort ${dataset}/${split}/${f} -o ${dataset}/${split}/${f}
            done

            utils/utt2spk_to_spk2utt.pl ${dataset}/${split}/utt2spk > ${dataset}/${split}/spk2utt

            dos2unix ${dataset}/${split}/text

            iconv -f utf-8 -t ascii//TRANSLIT  ${dataset}/${split}/text > ${dataset}/${split}/text.ascii
            mv ${dataset}/${split}/text.ascii ${dataset}/${split}/text

            utils/validate_data_dir.sh --no-feats ${dataset}/${split}
        done
        mv ${dataset} data
    done
    log "Stage 8: Creating test sets completed."
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
