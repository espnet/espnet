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
help_message=$(cat << EOF
Usage: $0

Options:
    --remove_archive (bool): true or false
      With remove_archive=True, the archives will be removed after being successfully downloaded and un-tarred.
EOF
)
SECONDS=0

# Data preparation related
data_url=https://drive.google.com/drive/folders/10MMe5o3-TM_bus8b9U-wsGWQcjJ3G9x-
remove_archive=false
download_opt=
# han-tw, tailo, tailo-num, poj
output_text=tailo

log "$0 $*"


. ./utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -gt 1 ]; then
  log "${help_message}"
  exit 2
fi

if [ -z "${TATVOL2}" ]; then
  log "Error: \$TATVOL2 is not set in db.sh."
  exit 2
fi


log "Download data to ${TATVOL2}"
if [ ! -d "${TATVOL2}" ]; then
    mkdir -p "${TATVOL2}"
fi
# To absolute path
TATVOL2=$(cd ${TATVOL2}; pwd)


log "Data Preparation"
train_dir=data/train
# dev_dir=data/dev
# test_dir=data/test

mkdir -p $train_dir
# mkdir -p $dev_dir
# mkdir -p $test_dir

python3 local/data_prep.py ${TATVOL2} $output_text

for dir in $train_dir ; do # $dev_dir $test_dir
  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
done

# python3 local/check_train_test_duplicate.py

# validate formats
utils/validate_data_dir.sh --no-feats data/train
# utils/validate_data_dir.sh --no-feats data/dev
# utils/validate_data_dir.sh --no-feats data/test

log "Successfully finished. [elapsed=${SECONDS}s]"