#!/usr/bin/env bash
set -e
set -u
set -o pipefail

stage=1
stop_stage=100
n_proc=8

trg_dir=data

. utils/parse_options.sh
. db.sh
. path.sh
. cmd.sh

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

echo $RATS

# general configuration
# see the following link to download the dataset : https://catalog.ldc.upenn.edu/LDC2021S08
if [ -z "${RATS}" ]; then
        log "Fill the value of 'RATS' of db.sh"
        exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Generate the eval trials"

    if [ ! -x /usr/bin/wget ]; then
        log "Cannot execute wget. wget is required for download."
        exit 3
    fi

    # generate RATS trial files
    log "Generate RATS eval protocol."
    python local/rats_trial_prep.py --data_dir ${RATS} --out rats_eval_trial.txt

    log "Stage 1, DONE."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2, Change into kaldi-style feature."
    mkdir -p ${trg_dir}/rats_test
    mkdir -p ${trg_dir}/rats_train

    python local/data_prep.py --src "${RATS}data/dev/" --dst "${trg_dir}/rats_test/"
    python local/data_prep.py --src "${RATS}data/train/" --dst "${trg_dir}/rats_train/"

    for f in wav.scp utt2spk spk2utt; do
        sort ${trg_dir}/rats_test/${f} -o ${trg_dir}/rats_test/${f}
        sort ${trg_dir}/rats_train/${f} -o ${trg_dir}/rats_train/${f}
    done

    # make test trial compatible with ESPnet.
    # src2spk : dictionary to map source information of file name to the speaker ID

    python local/convert_trial.py --trial rats_eval_trial.txt --scp ${trg_dir}/rats_test/wav.scp --src2spk ${trg_dir}/rats_test/src2spk --out ${trg_dir}/rats_test

    log "Stage 2, DONE."

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
