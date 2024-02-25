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
n_proc=8

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
    log "Root dir set to ${ASVSpoof_LA}"
    data_dir_prefix=${ASVSpoof_LA}
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
        wget -O "${data_dir_prefix}/LA.zip" https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y
    else
       log "LA.zip exists. Skip downloading ASVspoof LA.zip"
    fi

    # unzip LA.zip
    if [ ! -d "${data_dir_prefix}/LA" ]; then
        log "Unzipping LA.zip..."
        unzip "${data_dir_prefix}/LA.zip"
        rm "${data_dir_prefix}/LA.zip" # cleanup
    else
       log "LA exists. Skip unzipping ASVspoof LA.zip"
    fi
    log "Stage 1, DONE."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Protocol modification for conformity with ESPnet"

    # make new data dir where , LA_asv_eval
    if [ ! -d "${data_dir_prefix}/LA_asv_eval" ]; then
        mkdir "${data_dir_prefix}/LA_asv_eval"
        # Combine male and female eval speaker enrollment utterances to one new file
        cp "${data_dir_prefix}/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.female.trn.txt" "${data_dir_prefix}/LA_asv_eval/trn.txt"
        cat "${data_dir_prefix}/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.male.trn.txt" >> "${data_dir_prefix}/LA_asv_eval/trn.txt"

        # Make concatenated speaker enrollment utterances for (approximate) averaging of embeddings 
        python local/cat_spk_utt.py --in_dir "${data_dir_prefix}/LA/ASVspoof2019_LA_eval/flac" --in_file "${data_dir_prefix}/LA_asv_eval/trn.txt" --out_dir "${data_dir_prefix}/LA_asv_eval/flac"
        # Copy eval files to same dir with new concat files
        log "Making single dir for eval..."
        find "${data_dir_prefix}/LA/ASVspoof2019_LA_eval/flac/" -name "*.flac" -print0 | xargs -0 cp -t "${data_dir_prefix}/LA_asv_eval/flac/"
        # Make new protocol file
        python local/convert_protocol.py --in_file "${data_dir_prefix}/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt" --out_file "${data_dir_prefix}/LA_asv_eval/protocol.txt"

    else
       log "LA_asv_eval exists. Skipping protocol modification"
    fi
    log "Stage 2, DONE."    
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Making Kaldi style files and trials"

    if [ ! -d "${trg_dir}" ]; then
        log "Making Kaldi style files and making trials"

        mkdir -p data/test
        # make kaldi-style files for ASV dev and test
        python3 local/asv_data_prep.py --src "${data_dir_prefix}/LA_asv_eval/flac/" --dst "${trg_dir}/test"
        for f in wav.scp utt2spk spk2utt; do
            sort ${trg_dir}/test/${f} -o ${trg_dir}/test/${f}
        done
        utils/validate_data_dir.sh --no-feats --no-text "data/test" || exit 1

        # make test trial compatible with ESPnet
        log "Making the trial compatible with ESPnet"
        python local/convert_trial.py --trial "${data_dir_prefix}/LA_asv_eval/protocol.txt" --scp ${trg_dir}/test/wav.scp --out ${trg_dir}/test

    else
        log "${trg_dir} exists. Skip making Kaldi style files and trials"
    fi
    log "Stage 3, DONE."    
fi

log "Successfully finished. [elapsed=${SECONDS}s]"