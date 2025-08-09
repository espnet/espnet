#!/usr/bin/env bash
set -e
set -u
set -o pipefail

stage=1
stop_stage=100
n_proc=8

data_dir_prefix= # root dir to save datasets.

trg_dir=data

. utils/parse_options.sh
. db.sh
. path.sh
. cmd.sh

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}



if [ -z ${data_dir_prefix} ]; then
    log "Root dir for dataset not defined, setting to ${MAIN_ROOT}/egs2/cnceleb/spk1/downloads"
    data_dir_prefix=${MAIN_ROOT}/egs2/cnceleb/spk1/downloads
else
    log "Root dir set to ${data_dir_prefix}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Download Musan and RIR_NOISES for augmentation."

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

    # Make scp files
    for x in music noise speech; do
        find ${data_dir_prefix}/musan/${x} -iname "*.wav" > ${data_dir_prefix}/musan_${x}.scp
    done

    # Use small and medium rooms, leaving out largerooms.
    find ${data_dir_prefix}/RIRS_NOISES/simulated_rirs/mediumroom -iname "*.wav" > ${data_dir_prefix}/rirs.scp
    find ${data_dir_prefix}/RIRS_NOISES/simulated_rirs/smallroom -iname "*.wav" >> ${data_dir_prefix}/rirs.scp
    log "Stage 1, DONE."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Download CN-Celeb V1 and V2 datasets"

    mkdir -p "${data_dir_prefix}"

    # CN-Celeb v1
    if [ ! -f "${data_dir_prefix}/cn-celeb_v2.tar.gz" ]; then
        log "Downloading CN-Celeb V1..."
        wget -P ${data_dir_prefix} -c https://www.openslr.org/resources/82/cn-celeb_v2.tar.gz
    else
        log "CN-Celeb V1 archive already exists. Skipping download."
    fi

    # CN-Celeb v2
    if [ ! -f "${data_dir_prefix}/cn-celeb2_v2.tar.gz" ]; then
        log "Downloading CN-Celeb V2 split parts..."
        for part in a b c; do
            wget -P ${data_dir_prefix} -c "https://www.openslr.org/resources/82/cn-celeb2_v2.tar.gza${part}"
        done

        log "Combining CN-Celeb V2 split parts..."
        cat "${data_dir_prefix}"/cn-celeb2_v2.tar.gza* > "${data_dir_prefix}/cn-celeb2_v2.tar.gz"
    else
        log "CN-Celeb V2 archive already exists. Skipping download."
    fi

    # Extraction
    if [ ! -d "${data_dir_prefix}/CN-Celeb_flac" ]; then
        log "Extracting CN-Celeb V1..."
        tar -xzf "${data_dir_prefix}/cn-celeb_v2.tar.gz" -C "${data_dir_prefix}"
    else
        log "CN-Celeb V1 already extracted."
    fi

    if [ ! -d "${data_dir_prefix}/CN-Celeb2_flac" ]; then
        log "Extracting CN-Celeb V2..."
        tar -xzf "${data_dir_prefix}/cn-celeb2_v2.tar.gz" -C "${data_dir_prefix}"
    else
        log "CN-Celeb V2 already extracted."
    fi

    log "Stage 2 DONE."
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Convert CN-Celeb FLAC to WAV"

    cnceleb1_flac_dir=${data_dir_prefix}/CN-Celeb_flac
    cnceleb2_flac_dir=${data_dir_prefix}/CN-Celeb2_flac
    cnceleb1_wav_dir=${data_dir_prefix}/CN-Celeb_wav
    cnceleb2_wav_dir=${data_dir_prefix}/CN-Celeb2_wav

    # Convert CN-Celeb1
    if [ ! -d "${cnceleb1_wav_dir}" ]; then
        log "Converting CN-Celeb1 FLAC to WAV..."
        python local/flac2wav.py --dataset_dir "${cnceleb1_flac_dir}" --nj ${n_proc}
    else
        log "CN-Celeb1 already converted to WAV. Skipping."
    fi

    # Convert CN-Celeb2
    if [ ! -d "${cnceleb2_wav_dir}" ]; then
        log "Converting CN-Celeb2 FLAC to WAV..."
        python local/flac2wav.py --dataset_dir "${cnceleb2_flac_dir}" --nj ${n_proc}
    else
        log "CN-Celeb2 already converted to WAV. Skipping."
    fi

    log "Stage 3 DONE."
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Change into kaldi-style feature."
    mkdir -p ${trg_dir}/cnceleb1_dev
    mkdir -p ${trg_dir}/cnceleb2
    mkdir -p ${trg_dir}/cnceleb1_test
    mkdir -p ${trg_dir}/cnceleb1_enroll
    mkdir -p ${trg_dir}/cnceleb1_enroll_sep

    # Create kaldi-style data files: wav.scp, utt2spk, spk2utt
    python local/data_prep.py --src "${data_dir_prefix}/CN-Celeb_wav/data" --dst "${trg_dir}/cnceleb1_dev" \
    --spk "${data_dir_prefix}/CN-Celeb_flac/dev/dev.lst"
    python local/data_prep.py --src "${data_dir_prefix}/CN-Celeb2_wav/data" --dst "${trg_dir}/cnceleb2" \
    --spk "${data_dir_prefix}/CN-Celeb2_flac/spk.lst"
    python local/data_prep.py --src "${data_dir_prefix}/CN-Celeb_wav/eval/test" --dst "${trg_dir}/cnceleb1_test"
    python local/data_prep.py --src "${data_dir_prefix}/CN-Celeb_wav/eval/enroll" --dst "${trg_dir}/cnceleb1_enroll"

    # Deal with separate enroll utterances
    enroll_map="${data_dir_prefix}/CN-Celeb_wav/eval/lists/enroll.map"
    enroll_sep_utt="${trg_dir}/cnceleb1_enroll_sep/utt.lst"
    awk '{for (i=2; i<=NF; i++) print $i}' ${enroll_map} > ${enroll_sep_utt}
    python local/data_prep.py --src "${data_dir_prefix}/CN-Celeb_wav/data" --dst "${trg_dir}/cnceleb1_enroll_sep" \
    --utt "${enroll_sep_utt}"

    # Sort the data files
    for f in wav.scp utt2spk spk2utt; do
        sort ${trg_dir}/cnceleb1_dev/${f} -o ${trg_dir}/cnceleb1_dev/${f}
        sort ${trg_dir}/cnceleb2/${f} -o ${trg_dir}/cnceleb2/${f}
        sort ${trg_dir}/cnceleb1_test/${f} -o ${trg_dir}/cnceleb1_test/${f}
        sort ${trg_dir}/cnceleb1_enroll/${f} -o ${trg_dir}/cnceleb1_enroll/${f}
        sort ${trg_dir}/cnceleb1_enroll_sep/${f} -o ${trg_dir}/cnceleb1_enroll_sep/${f}
    done

    # Combine CN-Celeb 1 and 2 dev sets for combined training set
    mkdir -p ${trg_dir}/cnceleb_train
    for f in wav.scp utt2spk spk2utt; do
        cat ${trg_dir}/cnceleb1_dev/${f} >> ${trg_dir}/cnceleb_train/${f}
        cat ${trg_dir}/cnceleb2/${f} >> ${trg_dir}/cnceleb_train/${f}
        sort ${trg_dir}/cnceleb_train/${f} -o ${trg_dir}/cnceleb_train/${f}
    done

    # Combine CN-Celeb enroll and test sets for evaluation
    mkdir -p ${trg_dir}/cnceleb1_eval
    mkdir -p ${trg_dir}/cnceleb1_eval_sep
    for f in wav.scp utt2spk spk2utt; do
        if [ -f ${trg_dir}/cnceleb1_eval/${f} ]; then
            rm ${trg_dir}/cnceleb1_eval/${f}
        fi
        cat ${trg_dir}/cnceleb1_test/${f} >> ${trg_dir}/cnceleb1_eval/${f}
        cat ${trg_dir}/cnceleb1_enroll/${f} >> ${trg_dir}/cnceleb1_eval/${f}
        sort ${trg_dir}/cnceleb1_eval/${f} -o ${trg_dir}/cnceleb1_eval/${f}

        # For separate enroll utterances
        if [ -f ${trg_dir}/cnceleb1_eval_sep/${f} ]; then
            rm ${trg_dir}/cnceleb1_eval_sep/${f}
        fi
        cat ${trg_dir}/cnceleb1_test/${f} >> ${trg_dir}/cnceleb1_eval_sep/${f}
        cat ${trg_dir}/cnceleb1_enroll_sep/${f} >> ${trg_dir}/cnceleb1_eval_sep/${f}
        sort ${trg_dir}/cnceleb1_eval_sep/${f} -o ${trg_dir}/cnceleb1_eval_sep/${f}
    done

    # Make test trial compatible with ESPnet.
    python local/convert_trial.py \
    --trial "${data_dir_prefix}/CN-Celeb_flac/eval/lists/trials.lst" \
    --test_scp "${trg_dir}/cnceleb1_test/wav.scp" \
    --enroll_scp "${trg_dir}/cnceleb1_enroll/wav.scp" \
    --out "${trg_dir}/cnceleb1_eval"

    # Create new trial for separate enroll utterances
    python local/convert_trial.py \
    --trial "${data_dir_prefix}/CN-Celeb_flac/eval/lists/trials.lst" \
    --test_scp "${trg_dir}/cnceleb1_test/wav.scp" \
    --enroll_scp "${trg_dir}/cnceleb1_enroll_sep/wav.scp" \
    --enroll_map "${trg_dir}/cnceleb1_enroll_sep/spk2utt" \
    --out "${trg_dir}/cnceleb1_eval_sep"

    # Create valid trial from CN-Celeb 1 eval set
    mkdir -p ${trg_dir}/cnceleb1_valid
    python local/generate_trial.py \
    --enroll_dir data/cnceleb1_enroll \
    --test_dir data/cnceleb1_test \
    --out_dir data/cnceleb1_valid \
    --trials_per_enroll 200 \
    --target_ratio 0.3 \
    --seed 42

    # Create valid trial for separate enroll utterances
    mkdir -p ${trg_dir}/cnceleb1_valid_sep
    python local/generate_trial.py \
    --enroll_dir data/cnceleb1_enroll_sep \
    --test_dir data/cnceleb1_test \
    --out_dir data/cnceleb1_valid_sep \
    --trials_per_enroll 50 \
    --target_ratio 0.3 \
    --seed 42

    log "Stage 4, DONE."

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
