#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh
. ./cmd.sh
. ./db.sh

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
help_message=$(
  cat <<EOF
Usage: $0

Options:
    --remove_archive (bool): true or false
      With remove_archive=True, the archives will be removed after being successfully downloaded and un-tarred.
EOF
)
SECONDS=0

stage=1
stop_stage=100

data_url=www.openslr.org/resources/68
remove_archive=false
download_opt=

log "$0 $*"
. utils/parse_options.sh

if [ $# -gt 1 ]; then
  log "${help_message}"
  exit 2
fi

if "$remove_archive"; then
  download_opt="--remove-archive"
fi

if [ -z "${MAGICDATA}" ]; then
  log "Error: \$MAGICDATA is not set in db.sh."
  exit 2
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Download data to ${MAGICDATA}"
  if [ ! -d "${MAGICDATA}" ]; then
    mkdir -p "${MAGICDATA}"
  fi
  # To absolute path
  MAGICDATA=$(
    cd ${MAGICDATA}
    pwd
  )

  # download, untar, and:
  #   1. mkdir wav
  #   2. put train, test, dev under wav/
  #   3. copy train.scp, test.scp, dev.scp into wav/train/wav.scp, wav/test/wav.scp, wav/dev/wav.scp

  local/download_and_untar.sh ${download_opt} "${MAGICDATA}" "${data_url}" train_set
  local/download_and_untar.sh ${download_opt} "${MAGICDATA}" "${data_url}" dev_set
  local/download_and_untar.sh ${download_opt} "${MAGICDATA}" "${data_url}" test_set
  local/download_and_untar.sh ${download_opt} "${MAGICDATA}" "${data_url}" metadata

  for subset in train_set dev_set test_set; do
    if [ ! -f ${MAGICDATA}/$subset.complete ]; then
      echo "$0: data part ${MAGICDATA}/$subset was not successfully downloaded or extracted."
      exit 1
    fi
  done

  mkdir -p ${MAGICDATA}/wav
  for subset in train dev test; do
    if [ ! -f ${MAGICDATA}/wav/$subset/.complete ]; then
      mv ${MAGICDATA}/$subset ${MAGICDATA}/wav
    fi

    cp ${MAGICDATA}/$subset.scp ${MAGICDATA}/wav/${subset}/wav.scp
    touch ${MAGICDATA}/wav/${subset}/.complete
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  ### Task dependent. You have to make data the following preparation part by yourself.
  ### But you can utilize Kaldi recipes in most cases
  echo "stage 0: Data preparation"

  local/prepare_data.sh ${MAGICDATA}/wav/train data/local/train data/train || exit 1
  local/prepare_data.sh ${MAGICDATA}/wav/test data/local/test data/test || exit 1
  local/prepare_data.sh ${MAGICDATA}/wav/dev data/local/dev data/dev || exit 1

  # Normalize text to capital letters
  for x in train dev test; do
    mv data/${x}/text data/${x}/text.org
    paste <(cut -f 1 data/${x}/text.org) <(cut -f 2 data/${x}/text.org | tr '[:lower:]' '[:upper:]') \
      >data/${x}/text
    rm data/${x}/text.org
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Remove english tokens from text and store it in data/train_noeng/
  echo "Exclude train utterances with English tokens. "
  echo "Set \$train_set to train in run.sh if you don't want this"
  mkdir -p data/train_noeng
  cp data/train/{wav.scp,spk2utt,utt2spk} data/train_noeng
  awk 'eng=0;{for(i=2;i<=NF;i++)if($i ~ /^.*[A-Z]+.*$/)eng=1}{if(eng==0)print $0}' data/train/text >data/train_noeng/text
  utils/fix_data_dir.sh data/train_noeng/
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
