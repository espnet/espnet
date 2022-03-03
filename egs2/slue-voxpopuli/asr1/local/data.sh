#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
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
log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh
VOXPOPULI="/ocean/projects/cis210027p/skumar8/slue-toolkit/datasets/slue-voxpopuli"
if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${VOXPOPULI}" ]; then
    log "Fill the value of 'VOXPOPULI' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${VOXPOPULI}/LICENSE.txt" ]; then
	echo "stage 1: Download data to ${VOXPOPULI}"
    else
        log "stage 1: ${VOXPOPULI}/LICENSE.txt is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"

    # Uncomment this to convert .ogg to .wav files
    # mkdir -p ${VOXPOPULI}/fine-tune_wav
    # for file in ${VOXPOPULI}/fine-tune/*.ogg; do
    # 	new_name="$(basename ${file} .ogg)"
    # 	sox ${file} -t wav ${VOXPOPULI}/fine-tune_wav/${new_name}.wav
    # done

    # mkdir -p ${VOXPOPULI}/dev_wav
    # for file in ${VOXPOPULI}/dev/*.ogg; do
    # 	new_name="$(basename ${file} .ogg)"
    # 	sox ${file} -t wav ${VOXPOPULI}/dev_wav/${new_name}.wav
    # done

    # mkdir -p ${VOXPOPULI}/test_wav
    # for file in ${VOXPOPULI}/test/*.ogg; do
    # 	new_name="$(basename ${file} .ogg)"
    # 	sox ${file} -t wav ${VOXPOPULI}/test_wav/${new_name}.wav
    # done

    mkdir -p data/{train,devel,test}
    # python3 local/data_prep_slue_entity.py  ${VOXPOPULI}
    python3 local/data_prep_original_slue_format.py  ${VOXPOPULI}
    for x in test devel train; do
        for f in text wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
        utils/validate_data_dir.sh --no-feats data/${x} || exit 1
    done
    local/run_spm.sh
    mv data data_old
    mv data_bpe_1000 data
  #   python local/prepare_entity_type.py
  #   for x in test devel train; do
  #       mv data/${x}/text data/${x}/text_old
	# mv data/${x}/text_new data/${x}/text
  #   done

    utils/validate_data_dir.sh --no-feats data/train
    utils/validate_data_dir.sh --no-feats data/devel
    utils/validate_data_dir.sh --no-feats data/test

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
