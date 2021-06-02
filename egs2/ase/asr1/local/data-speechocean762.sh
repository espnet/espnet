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
data=/home/storage07/zhangjunbo/data


log "$0 $*"
# . utils/parse_options.sh

# . ./db.sh
# . ./path.sh
# . ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  for part in train test; do
    local/prep-speechocean762.sh $data/speechocean762/$part data/$part
  done

  mkdir -p data/local
  cp $data/speechocean762/resource/* data/local
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  text_phone="data/local/text-phone"
  utt2phone="data/local/utt2phone"
  python local/get_utt2phone.py ${text_phone} ${utt2phone}

  for part in train test; do
    cp ${utt2phone} data/$part/text
  done
fi

for part in train test; do
  utils/fix_data_dir.sh data/$part || exit 1;
done

log "Successfully finished. [elapsed=${SECONDS}s]"
