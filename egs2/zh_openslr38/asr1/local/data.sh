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
data_url=www.openslr.org/resources/38
remove_archive=false
download_opt=

log "$0 $*"


. ./utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -gt 1 ]; then
  log "${help_message}"
  exit 2
fi

if "$remove_archive"; then
  download_opt="--remove-archive"
fi

if [ -z "${ST_CMDS}" ]; then
  log "Error: \$ST_CMDS is not set in db.sh."
  exit 2
fi


log "Download data to ${ST_CMDS}"
if [ ! -d "${ST_CMDS}" ]; then
    mkdir -p "${ST_CMDS}"
fi
# To absolute path
ST_CMDS=$(cd ${ST_CMDS}; pwd)

echo local/data_download.sh ${download_opt} "${ST_CMDS}" "${data_url}" ST-CMDS-20170001_1-OS.tar.gz
local/data_download.sh ${download_opt} "${ST_CMDS}" "${data_url}" ST-CMDS-20170001_1-OS.tar.gz

log "Data Preparation"
train_dir=data/train
dev_dir=data/dev
test_dir=data/test

mkdir -p $train_dir
mkdir -p $dev_dir
mkdir -p $test_dir

python3 local/data_split.py ${ST_CMDS}/ST-CMDS-20170001_1-OS

for dir in $train_dir $dev_dir $test_dir; do
  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
done

python3 local/check_train_test_duplicate.py

# validate formats
utils/validate_data_dir.sh --no-feats data/train
utils/validate_data_dir.sh --no-feats data/dev
utils/validate_data_dir.sh --no-feats data/test

log "Successfully finished. [elapsed=${SECONDS}s]"
