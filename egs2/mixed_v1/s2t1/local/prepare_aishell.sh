#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh || exit 1;

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

data_dir=/scratch/bbjs/peng6/corpora/AISHELL-1
prefix=AISHELL-1
output_dir=data/AISHELL-1
splits="dev train"

python local/prepare_aishell.py \
    --data_dir ${data_dir} \
    --prefix ${prefix} \
    --output_dir ${output_dir} \
    --splits ${splits} || exit 1;

utt_extra_files="text.prev text.ctc"
for x in ${splits}; do
    # NOTE(yifan): extra text files must be sorted and unique
    for f in ${utt_extra_files}; do
        check_sorted ${output_dir}/${x}/${f}
    done
    utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${output_dir}/${x} || exit 1;
    utils/validate_data_dir.sh --no-feats --non-print ${output_dir}/${x} || exit 1;
done

log "Successfully finished. [elapsed=${SECONDS}s]"
