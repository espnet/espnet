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

    if [ ! -d "${trg_dir}/asvspoof5_train" ]; then
        log "Making Kaldi style files for train"
        mkdir -p "${trg_dir}/asvspoof5_train"
        python3 local/asvspoof5_train_data_prep.py "${data_dir_prefix}/asvspoof5_data" "${trg_dir}/asvspoof5_train"
        for f in wav.scp utt2spk utt2spf; do
            sort ${trg_dir}/asvspoof5_train/${f} -o ${trg_dir}/asvspoof5_train/${f}
        done
        utils/utt2spk_to_spk2utt.pl ${trg_dir}/asvspoof5_train/utt2spk > "${trg_dir}/asvspoof5_train/spk2utt"
        utils/utt2spk_to_spk2utt.pl ${trg_dir}/asvspoof5_train/utt2spf > "${trg_dir}/asvspoof5_train/spf2utt"
        utils/validate_data_dir.sh --no-feats --no-text "${trg_dir}/asvspoof5_train" || exit 1
    else
        log "${trg_dir}/asvspoof5_train exists. Skip making Kaldi style files for train"
    fi

    log "Stage 1, DONE."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation for dev and eval"

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

        # copy enrollment file
        cp "${data_dir_prefix}/asvspoof5_data/ASVspoof5.dev.enroll.txt" "${trg_dir}/dev/enroll.txt"

        # make trials for dev set
        log "Making the dev trial compatible with ESPnet"
        python local/convert_trial.py --trial "${data_dir_prefix}/asvspoof5_data/ASVspoof5.dev.trial.txt" --enroll ${trg_dir}/dev/enroll.txt  --scp ${trg_dir}/dev/wav.scp --out ${trg_dir}/dev --task dev
        # sort files
        for x in "trial.scp" "trial2.scp" "trial3.scp" "trial4.scp" "trial_label"; do
            sort ${trg_dir}/dev/${x} -o ${trg_dir}/dev/${x}
        done
    else
        log "${trg_dir}/dev exists. Skip making Kaldi style files for dev"
    fi

    if [ ! -d "${trg_dir}/eval" ]; then
        log "Making Kaldi style files for eval"
        mkdir -p "${trg_dir}/eval"

        log "Making Kaldi style files for eval"
        python3 local/eval_data_prep.py --asvspoof5_root "${data_dir_prefix}/asvspoof5_data" --target_dir "${trg_dir}/eval" --is_progress True

        for f in wav.scp utt2spk utt2spf; do
            sort ${trg_dir}/eval/${f} -o ${trg_dir}/eval/${f}
        done
        utils/utt2spk_to_spk2utt.pl ${trg_dir}/eval/utt2spk > "${trg_dir}/eval/spk2utt"
        utils/utt2spk_to_spk2utt.pl ${trg_dir}/eval/utt2spf > "${trg_dir}/eval/spf2utt"
        utils/validate_data_dir.sh --no-feats --no-text "${trg_dir}/eval" || exit 1

        # copy enrollment file
        cp "${data_dir_prefix}/asvspoof5_data/ASVspoof5.track_2.progress.enroll.txt" "${trg_dir}/eval/enroll.txt"

        # make trials for eval set
        log "Making the eval trial compatible with ESPnet"
        python local/convert_trial.py --trial "${data_dir_prefix}/asvspoof5_data/ASVspoof5.track_2.progress.trial.txt" --enroll ${trg_dir}/eval/enroll.txt  --scp ${trg_dir}/eval/wav.scp --out ${trg_dir}/eval --task eval

    else
        log "${trg_dir}/eval exists. Skip making Kaldi style files for eval"
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

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Generate UTMOS teacher model data"

    utmos_pseudomos_dir="${data_dir_prefix}/spk1/data/UTMOS_pseudomos"
    log "Computing UTMOS pseudomos scores"

    for x in "asvspoof5_train" "dev"; do
        if [ ! -d "${utmos_pseudomos_dir}/${x}" ]; then
            mkdir -p ${utmos_pseudomos_dir}/${x}
            python3 pyscripts/utils/evaluate_pseudomos.py "${data_dir_prefix}/spk1/data/${x}/wav.scp" --outdir ${utmos_pseudomos_dir}/${x} --batchsize 4

        else
            log "${utmos_pseudomos_dir}/${x} exists. Skip computing UTMOS pseudomos scores."
        fi
    done

    for x in "asvspoof5_train" "dev"; do
        if [ ! -f "${trg_dir}/${x}/utt2pmos" ]; then
            log "Creating utt2pmos file for ${x}"
            python3 local/convert_pmos_uttids.py --in_utt2pmos ${utmos_pseudomos_dir}/${x}/utt2pmos --wavscp ${trg_dir}/${x}/wav.scp --out  ${trg_dir}/${x}
        else
            log "utt2pmos file for ${x} exists. Skip creating it."
        fi

        # sort files
        sort ${trg_dir}/${x}/utt2pmos -o ${trg_dir}/${x}/utt2pmos
    done

    for x in "asvspoof5_train" "dev"; do
        if [ ! -d "${trg_dir}/${x}/frame_features" ]; then
            log "Computing frame features for ${x}"
            python3 local/eval_pmos.py "${data_dir_prefix}/spk1/data/${x}/wav.scp" --outdir ${trg_dir}/${x} --batchsize 4
        else
            log "Frame features for ${x} exists. Skip computing it."
        fi
    done

    # make feats.scp for train
    if [ ! -f "${trg_dir}/asvspoof5_train/feats.scp" ]; then
        log "Creating feats.scp for asvspoof5_train"
        python3 local/make_feats_scp.py --input_dir ${trg_dir}/asvspoof5_train/frame_features --output_file ${trg_dir}/asvspoof5_train/feats.scp
        # sort files
        sort ${trg_dir}/asvspoof5_train/feats.scp -o ${trg_dir}/asvspoof5_train/feats.scp
    else
        log "feats.scp for asvspoof5_train exists. Skip creating it."
    fi

    # make feats.scp for dev
    if [ ! -f "${trg_dir}/dev/feats.scp" ]; then
        log "Creating feats.scp for dev"
        python3 local/make_feats_scp.py --input_dir ${trg_dir}/dev/frame_features --output_dir ${trg_dir}/dev --trial "${data_dir_prefix}/asvspoof5_data/ASVspoof5.dev.trial.txt" --enroll ${trg_dir}/dev/enroll.txt --task dev
        # sort files
        for x in "feats" "feats1" "feats2" "feats3" "feats4"; do
            sort ${trg_dir}/dev/${x}.scp -o ${trg_dir}/dev/${x}.scp
        done
    else
        log "feats.scp for dev exists. Skip creating it."
    fi

    log "Stage 4, DONE."
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
