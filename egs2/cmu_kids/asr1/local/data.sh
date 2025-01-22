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
sph2wav=true


log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${CMU_KIDS}" ]; then
    log "Fill the value of 'CMU_KIDS' in db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -d "${CMU_KIDS}" ]; then
        log "stage 1: Please download data from https://catalog.ldc.upenn.edu/LDC97S63 and save to ${CMU_KIDS}"
        exit 1
    else
        log "stage 1: ${CMU_KIDS} already exists. Skipping data downloading."
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    original_dir="${CMU_KIDS}"
    data_dir="./data"
    lists_dir="conf/file_list"

    local/cmu_kids_data_prepare.sh $original_dir/ $data_dir/ $lists_dir/
    log "stage 2: Data preparation completed."
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Wav conversion"

    if $sph2wav; then
        log "Converting .sph to .wav"

        # Loop over each split (train, dev, test)
        for x in train dev test; do
            mkdir -p data/${x}/wav
            # Initialize a new wav.scp file
            true > data/${x}/wav.scp.new

            # Process each utterance in the wav.scp file
            while IFS=' ' read -r uttID wavCmd; do
                # Extract the .sph file path from wavCmd
                sphFile=$(echo $wavCmd | sed -e 's/.*sph2pipe -f wav -p -c 1 //; s/|$//')

                # Define the new .wav file path
                wavFile="data/${x}/wav/${uttID}.wav"

                # Convert .sph to .wav
                python local/sph2wav.py --input $sphFile --output $wavFile

                # Write the new entry to the wav.scp file
                echo "$uttID $wavFile" >> data/${x}/wav.scp.new
            done < data/${x}/wav.scp
            # Clean up old wav.scp
            rm data/${x}/wav.scp
            # Rename the new wav.scp
            mv data/${x}/wav.scp.new data/${x}/wav.scp

        done
        log "Stage 3: Finished .sph to .wav conversion"
    else
        log "Stage 3: .sph to .wav conversion skipped."
    fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
