#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./db.sh || exit 1;

stage=1
stop_stage=2

. utils/parse_options.sh || exit 1;
# Copied from utils/fix_data_dir.sh
function check_sorted {
  file=$1
  sort -k1,1 -u <$file >$file.tmp
  if ! cmp -s $file $file.tmp; then
    echo "$0: file $1 is not in sorted order or not unique, sorting it"
    mv $file.tmp $file
  else
    rm $file.tmp
  fi
}

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

prefix=FLEURS
output_dir=data/FLEURS
splits="train valid test"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python local/prepare_fleurs.py \
        --lang all \
        --cache ${FLEURS} \
        --prefix FLEURS \
        --output_dir ${output_dir} || exit 1;
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    utt_extra_files="text.prev text.ctc"
    for x in ${splits}; do
        # NOTE(yifan): extra text files must be sorted and unique
        for f in ${utt_extra_files}; do
            check_sorted ${output_dir}/${x}/${f}
        done
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${output_dir}/${x} || exit 1;
        utils/validate_data_dir.sh --no-feats --non-print ${output_dir}/${x} || exit 1;
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
