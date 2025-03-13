#!/bin/bash
#SBATCH --job-name=watkins_ssl
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=70
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.watkins_ssl

mkdir -p logs

. ./db.sh
. ./path.sh

PARALLELISM=64
echo $(which python)

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

log "$0 $*"
. utils/parse_options.sh

. ./path.sh
. ./cmd.sh

# Parse arguments
WRITE_DIR=/work/nvme/bbjs/sbharadwaj/watkins_ssl/data
READ_DIR=/work/hdd/bbjs/shared/corpora/watkins_marine/scraped_data
# master_tapes 
for set in cut_tapes; do
    mkdir -p ${WRITE_DIR}/${set}
    log "Processing Watkins dataset from ${READ_DIR}/${set}"
    python3 local/data_prep_watkins_ssl.py ${READ_DIR}/${set} ${WRITE_DIR}/${set}
    for f in wav.scp utt2spk segments; do
        if [ ! -f "${WRITE_DIR}/${set}/${f}" ]; then
            log "Error: ${WRITE_DIR}/${set}/${f} does not exist"
            exit 1
        fi
        sort ${WRITE_DIR}/${set}/${f} -o ${WRITE_DIR}/${set}/${f}
    done
    utils/utt2spk_to_spk2utt.pl ${WRITE_DIR}/${set}/utt2spk > "${WRITE_DIR}/${set}/spk2utt"
    utils/validate_data_dir.sh --no-text --no-feats ${WRITE_DIR}/${set} || exit 1
done

# Prepare cut_tapes with additional noise
mkdir -p ${WRITE_DIR}/cut_tapes_noise
log "Processing Watkins dataset from ${READ_DIR}/cut_tapes"
python3 local/data_prep_watkins_ssl.py ${READ_DIR}/cut_tapes ${WRITE_DIR}/cut_tapes_noise --add_noise
for f in wav.scp utt2spk segments; do
    if [ ! -f "${WRITE_DIR}/cut_tapes_noise/${f}" ]; then
        log "Error: ${WRITE_DIR}/cut_tapes_noise/${f} does not exist"
        exit 1
    fi
    sort ${WRITE_DIR}/cut_tapes_noise/${f} -o ${WRITE_DIR}/cut_tapes_noise/${f}
done
utils/utt2spk_to_spk2utt.pl ${WRITE_DIR}/cut_tapes_noise/utt2spk > "${WRITE_DIR}/cut_tapes_noise/spk2utt"
utils/validate_data_dir.sh --no-text --no-feats ${WRITE_DIR}/cut_tapes_noise || exit 1


log "Successfully finished processing Watkins dataset. [elapsed=${SECONDS}s]"
