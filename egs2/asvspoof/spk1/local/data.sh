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
    log "Root dir for dataset not defined, setting to ${MAIN_ROOT}/egs2/asvspoof"
    data_dir_prefix=${MAIN_ROOT}/egs2/asvspoof
else
    log "Root dir set to ${ASVSPOOF2019_LA}"
    data_dir_prefix=${ASVSPOOF2019_LA}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Download ASVspoof LA.zip"
    if [ ! -x /usr/bin/wget ]; then
        log "Cannot execute wget. wget is required for download."
        exit 3
    fi

    # download ASVspoof LA.zip
    if [ ! -f "${data_dir_prefix}/LA.zip" ]; then
        log "Downloading ASVspoof LA.zip..."
        wget -O "${data_dir_prefix}/LA.zip" "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"
    else
       log "LA.zip exists. Skip downloading ASVspoof LA.zip"
    fi

    # unzip LA.zip
    if [ ! -d "${data_dir_prefix}/LA" ]; then
        log "Unzipping LA.zip..."
        unzip "${data_dir_prefix}/LA.zip" -d "${data_dir_prefix}"
    else
       log "LA exists. Skip unzipping ASVspoof LA.zip"
    fi
    log "Stage 1, DONE."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Protocol modification for conformity with ESPnet"

    # make new data dirs, LA_asv_eval and LA_asv_dev with concatenated enrollments
    for x in eval dev; do
        if [ ! -d "${data_dir_prefix}/LA_asv_${x}" ]; then
            mkdir "${data_dir_prefix}/LA_asv_${x}"
            # Combine male and female eval/dev speaker enrollment utterances to one new file
            cp "${data_dir_prefix}/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.${x}.female.trn.txt" "${data_dir_prefix}/LA_asv_${x}/trn.txt"
            cat "${data_dir_prefix}/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.${x}.male.trn.txt" >> "${data_dir_prefix}/LA_asv_${x}/trn.txt"

            # Make concatenated speaker enrollment utterances for (approximate) averaging of embeddings
            python local/cat_spk_utt.py --in_dir "${data_dir_prefix}/LA/ASVspoof2019_LA_${x}/flac" --in_file "${data_dir_prefix}/LA_asv_${x}/trn.txt" --out_dir "${data_dir_prefix}/LA_asv_${x}/flac"
            # Copy eval/dev files to same dir with new concat files
            log "Making single dir for ${x}..."
            find "${data_dir_prefix}/LA/ASVspoof2019_LA_${x}/flac/" -name "*.flac" -print0 | xargs -0 cp -t "${data_dir_prefix}/LA_asv_${x}/flac/"
            # Make new protocol file
            python local/convert_protocol.py --in_file "${data_dir_prefix}/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.${x}.gi.trl.txt" --out_file "${data_dir_prefix}/LA_asv_${x}/protocol.txt"

        else
            log "LA_asv_${x} exists. Skipping protocol modification"
        fi

    done

    log "Stage 2, DONE."
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Making Kaldi style files and trials"

    if [ ! -d "${trg_dir}" ]; then
        log "Making Kaldi style files and making trials"

        mkdir -p data/test
        mkdir -p data/dev
        # make kaldi-style files for ASV test/dev
        python3 local/asv_data_prep.py --src "${data_dir_prefix}/LA_asv_eval/flac/" --dst "${trg_dir}/test" --partition "eval"
        python3 local/asv_data_prep.py --src "${data_dir_prefix}/LA_asv_dev/flac/" --dst "${trg_dir}/dev" --partition "dev"
        for f in wav.scp utt2spk spk2utt; do
            sort ${trg_dir}/test/${f} -o ${trg_dir}/test/${f}
            sort ${trg_dir}/dev/${f} -o ${trg_dir}/dev/${f}
        done
        utils/validate_data_dir.sh --no-feats --no-text "data/test" || exit 1
        utils/validate_data_dir.sh --no-feats --no-text "data/dev" || exit 1

        # make test trial compatible with ESPnet
        log "Making the trial compatible with ESPnet"
        python local/convert_trial.py --trial "${data_dir_prefix}/LA_asv_eval/protocol.txt" --scp ${trg_dir}/test/wav.scp --out ${trg_dir}/test
        python local/convert_trial.py --trial "${data_dir_prefix}/LA_asv_dev/protocol.txt" --scp ${trg_dir}/dev/wav.scp --out ${trg_dir}/dev

    else
        log "${trg_dir} exists. Skip making Kaldi style files and trials"
    fi
    log "Stage 3, DONE."
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Data Preparation for train"
    mkdir -p data/train
    python3 local/cm_data_prep.py ${data_dir_prefix}
    for f in wav.scp utt2spk; do
        sort data/train/${f} -o data/train/${f}
    done
    utils/utt2spk_to_spk2utt.pl data/train/utt2spk > "data/train/spk2utt"
    utils/validate_data_dir.sh --no-feats --no-text data/train || exit 1
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Download Musan and RIR_NOISES for augmentation."

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
    log "Stage 5, DONE."
fi

# types of trials:                            # num of examples
# SV: bonafide target * bonafide nontarget    5370 | 33327
# A07: bonafide target * A07 spoof            5370 | 4914
# A08: bonafide target * A08 spoof            5370 | 4914
# A09: bonafide target * A09 spoof            5370 | 4914
# A10: bonafide target * A10 spoof            5370 | 4914
# A11: bonafide target * A11 spoof            5370 | 4914
# A12: bonafide target * A12 spoof            5370 | 4914
# A13: bonafide target * A13 spoof            5370 | 4914
# A14: bonafide target * A14 spoof            5370 | 4914
# A15: bonafide target * A15 spoof            5370 | 4914
# A16: bonafide target * A16 spoof            5370 | 4914
# A17: bonafide target * A17 spoof            5370 | 4914
# A18: bonafide target * A18 spoof            5370 | 4914
# A19: bonafide target * A19 spoof            5370 | 4914
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "stage 6: Making sub-protocols and Kaldi-style files for each subset of trials"

    log "Making sub-protocols..."
    if [ ! -f "${data_dir_prefix}/SV.txt" ]; then
        for x in SV A07 A08 A09 A10 A11 A12 A13 A14 A15 A16 A17 A18 A19; do
            # Make new protocol file
            python local/make_subprotocols.py --in_file "${data_dir_prefix}/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt" --out_file "${data_dir_prefix}/LA_asv_eval/${x}.txt" --trial_type ${x}
        done
    else
       log "Sub protocol files exist. Skip making sub-protocols"
    fi

    log "Making Kaldi-style files for each subprotocol"
    if [ ! -d "${trg_dir}/test_SV" ]; then
        log "Making Kaldi style files and making trials"
        for x in SV A07 A08 A09 A10 A11 A12 A13 A14 A15 A16 A17 A18 A19; do
            mkdir -p data/test_${x}
            # make kaldi-style files for ASV dev and test
            python3 local/asv_data_prep.py --src "${data_dir_prefix}/LA_asv_eval/flac/" --dst "${trg_dir}/test_${x}"
            for f in wav.scp utt2spk spk2utt; do
                sort ${trg_dir}/test_${x}/${f} -o ${trg_dir}/test_${x}/${f}
            done
            utils/validate_data_dir.sh --no-feats --no-text "data/test_${x}" || exit 1

            # make test trial compatible with ESPnet
            log "Making the trial compatible with ESPnet"
            python local/convert_trial.py --trial "${data_dir_prefix}/LA_asv_eval/${x}.txt" --scp ${trg_dir}/test_${x}/wav.scp --out ${trg_dir}/test_${x}
        done
    else
        log "${trg_dir}/test_SV exists. Skip making Kaldi style files and trials"
    fi

    log "Stage 6, DONE."
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
