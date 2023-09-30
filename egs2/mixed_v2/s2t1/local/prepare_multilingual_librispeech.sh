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

mls_dir=/scratch/bbjs/peng6/corpora/multilingual_librispeech
prefix=MLS
output_dir=data/${prefix}
# languages="nl fr de it pl pt es en"
languages="en"
splits="dev train"

utt_extra_files="text.prev text.ctc"

for lang in ${languages}; do
    python local/prepare_multilingual_librispeech.py \
        --data_dir ${mls_dir} \
        --prefix ${prefix} \
        --output_dir ${output_dir} \
        --splits ${splits} \
        --langs ${lang} || exit 1;

        for x in ${splits}; do
            # NOTE(yifan): extra text files must be sorted and unique
            for f in ${utt_extra_files}; do
                check_sorted ${output_dir}/${x}.${lang}/${f}
            done
            utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${output_dir}/${x}.${lang} || exit 1;
            utils/validate_data_dir.sh --no-feats --non-print ${output_dir}/${x}.${lang} || exit 1;
        done
done

log "Successfully finished. [elapsed=${SECONDS}s]"
