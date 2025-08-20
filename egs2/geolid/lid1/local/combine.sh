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

if [ -d dump/raw/train_babel_over_10s_lang ]; then
  log "Directory dump/raw/train_babel_over_10s_lang exists."
else
  log "Directory dump/raw/train_babel_over_10s_lang does not exist. Running local/filter_babel_train.sh."
  local/filter_babel_train.sh || exit 1
fi


train_sets="dump/raw/train_fleurs_lang \
            dump/raw/train_ml_superb2_lang \
            dump/raw/train_voxlingua107_lang \
            dump/raw/train_voxpopuli_lang \
            dump/raw/train_babel_over_10s_lang \
            "

train_out=dump/raw/train_all_no_filter_lang

mkdir -p ${train_out}

# Combine train
# do not combine the segments
mv dump/raw/train_babel_lang/segments dump/raw/train_babel_lang/segments.backup
utils/combine_data.sh --skip_fix true ${train_out} ${train_sets} || exit 1;
mv dump/raw/train_babel_lang/segments.backup dump/raw/train_babel_lang/segments

python local/create_utt2dataset.py --train_sets "${train_sets}"

utils/utt2spk_to_spk2utt.pl ${train_out}/utt2lang > ${train_out}/lang2utt || exit 1;
utils/utt2spk_to_spk2utt.pl ${train_out}/utt2dataset > ${train_out}/dataset2utt || exit 1;

cp ${train_out}/lang2utt ${train_out}/category2utt || exit 1;

utils/fix_data_dir.sh ${train_out} || exit 1;
utils/validate_data_dir.sh --no-feats --non-print --no-text ${train_out} || exit 1;

log "Successfully finished. [elapsed=${SECONDS}s]"
