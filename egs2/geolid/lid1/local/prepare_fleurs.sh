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

output_dir=data
splits="train_fleurs_lang dev_fleurs_lang test_fleurs_lang"

mkdir -p ${output_dir}
for x in ${splits}; do
  mkdir -p ${output_dir}/${x}
done

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python local/prepare_fleurs.py \
      --output_dir ${output_dir} || exit 1;
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    for x in ${splits}; do
      check_sorted ${output_dir}/${x}/utt2lang
      check_sorted ${output_dir}/${x}/wav.scp
      ./utils/utt2spk_to_spk2utt.pl ${output_dir}/${x}/utt2lang > ${output_dir}/${x}/lang2utt
      cp ${output_dir}/${x}/lang2utt ${output_dir}/${x}/category2utt
      mv ${output_dir}/${x}/utt2lang ${output_dir}/${x}/utt2spk
      mv ${output_dir}/${x}/lang2utt ${output_dir}/${x}/spk2utt
      utils/fix_data_dir.sh ${output_dir}/${x} || exit 1;
      utils/validate_data_dir.sh --no-feats --non-print --no-text ${output_dir}/${x} || exit 1;
      mv ${output_dir}/${x}/utt2spk ${output_dir}/${x}/utt2lang
      mv ${output_dir}/${x}/spk2utt ${output_dir}/${x}/lang2utt
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
