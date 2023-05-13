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

old_data=data/CoVoST2.old
new_data=data/CoVoST2
splits="dev train"

for dset in ${splits}; do
    mkdir -p ${new_data}/${dset}

    python3 local/filter_covost2.py \
        --input ${old_data}/${dset} \
        --output ${new_data}/${dset}
done

cp ${old_data}/nlsyms.txt ${new_data}

utt_extra_files="text.prev text.ctc"
for x in ${splits}; do
    # NOTE(yifan): extra text files must be sorted and unique
    for f in ${utt_extra_files}; do
        check_sorted ${new_data}/${x}/${f}
    done
    utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${new_data}/${x} || exit 1;
    utils/validate_data_dir.sh --no-feats --non-print ${new_data}/${x} || exit 1;
done

log "Successfully finished. [elapsed=${SECONDS}s]"
