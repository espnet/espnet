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

data_dir_prefix= # root dir to save datasets
trg_dir=data


log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z ${data_dir_prefix} ]; then
    log "Root dir for dataset not defined, setting to ${MAIN_ROOT}/egs2/asvspoof5"
    data_dir_prefix=${MAIN_ROOT}/egs2/asvspoof5
else
    log "Root dir set to ${ASVSPOOF5}"
    data_dir_prefix=${ASVSPOOF5}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Preparation for train"
    
    if [ ! -d "${trg_dir}/train" ]; then
        log "Making Kaldi style files for train"
        mkdir -p "${trg_dir}/train"
        python3 local/train_data_prep.py "${data_dir_prefix}/asvspoof5_data" "${trg_dir}/train"
        for f in wav.scp utt2spk utt2spf; do
            sort ${trg_dir}/train/${f} -o ${trg_dir}/train/${f}
        done
        utils/utt2spk_to_spk2utt.pl ${trg_dir}/train/utt2spk > "${trg_dir}/train/spk2utt"
        utils/utt2spk_to_spk2utt.pl ${trg_dir}/train/utt2spf > "${trg_dir}/train/spf2utt"
        utils/validate_data_dir.sh --no-feats --no-text "${trg_dir}/train" || exit 1
    else
        log "${trg_dir}/train exists. Skip making Kaldi style files for train"
    fi

    log "Stage 1, DONE."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation for dev"
    # concatenate the utterances for dev set enrollment
    if [ ! -d "${data_dir_prefix}/asvspoof5_data/cat_flac_D" ]; then
        log "Making concatenated utterances for dev set enrollment"
        mkdir -p "${data_dir_prefix}/asvspoof5_data/cat_flac_D"
        python local/cat_utt.py --in_dir "${data_dir_prefix}/asvspoof5_data/flac_D" --enrollment_file "${data_dir_prefix}/asvspoof5_data/ASVspoof5.dev.enroll.txt" --out_dir "${data_dir_prefix}/asvspoof5_data/cat_flac_D"
    else
        log "${data_dir_prefix}/asvspoof5_data/cat_flac_D exists. Skip making concatenated utterances for dev"
    fi

    if [ ! -d "${trg_dir}/dev" ]; then
        log "Making Kaldi style files for dev"
        mkdir -p "${trg_dir}/dev"
        python3 local/dev_data_prep.py --asvspoof5_root "${data_dir_prefix}/asvspoof5_data" --target_dir "${trg_dir}/dev"
        for f in wav.scp utt2spk utt2spf; do
            sort ${trg_dir}/dev/${f} -o ${trg_dir}/dev/${f}
        done
        utils/utt2spk_to_spk2utt.pl ${trg_dir}/dev/utt2spk > "${trg_dir}/dev/spk2utt"
        utils/utt2spk_to_spk2utt.pl ${trg_dir}/dev/utt2spf > "${trg_dir}/dev/spf2utt"
        utils/validate_data_dir.sh --no-feats --no-text "${trg_dir}/dev" || exit 1

        # make trials for dev set
        log "Making the trial compatible with ESPnet"
        python local/convert_trial.py --trial "${data_dir_prefix}/asvspoof5_data/ASVspoof5.dev.trial.txt" --scp ${trg_dir}/dev/wav.scp --out ${trg_dir}/dev
    else
        log "${trg_dir}/dev exists. Skip making Kaldi style files for dev"
    fi

    log "Stage 2, DONE."
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Download Musan and RIR_NOISES for augmentation."

    if [ ! -f ${data_dir_prefix}/rirs_noises.zip ]; then
        wget -P ${data_dir_prefix} -c http://www.openslr.org/resources/28/rirs_noises.zip
    else
        log "RIRS_NOISES exists. Skip download."
    fi

    if [ ! -f ${data_dir_prefix}/musan.tar.gz ]; then
        wget -P ${data_dir_prefix} -c http://www.openslr.org/resources/17/musan.tar.gz
    else
        log "Musan exists. Skip download."
    fi

    if [ -d ${data_dir_prefix}/RIRS_NOISES ]; then
        log "Skip extracting RIRS_NOISES"
    else
        log "Extracting RIR augmentation data."
        unzip -q ${data_dir_prefix}/rirs_noises.zip -d ${data_dir_prefix}
    fi

    if [ -d ${data_dir_prefix}/musan ]; then
        log "Skip extracting Musan"
    else
        log "Extracting Musan noise augmentation data."
        tar -zxvf ${data_dir_prefix}/musan.tar.gz -C ${data_dir_prefix}
    fi

    # make scp files
    log "Making scp files for musan"
    for x in music noise speech; do
        find ${data_dir_prefix}/musan/${x} -iname "*.wav" > ${trg_dir}/musan_${x}.scp
    done

    # Use small and medium rooms, leaving out largerooms.
    # Similar setup to Kaldi and VoxCeleb_trainer.
    log "Making scp files for RIRS_NOISES"
    find ${data_dir_prefix}/RIRS_NOISES/simulated_rirs/mediumroom -iname "*.wav" > ${trg_dir}/rirs.scp
    find ${data_dir_prefix}/RIRS_NOISES/simulated_rirs/smallroom -iname "*.wav" >> ${trg_dir}/rirs.scp
    log "Stage 3, DONE."
fi

# types of trials:                            # num of examples

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Making sub-protocols and Kaldi-style files for each subset of trials"

    log "Stage 4, DONE."
fi

log "Successfully finished. [elapsed=${SECONDS}s]"