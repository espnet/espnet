#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


. ./path.sh || exit 1;
. ./db.sh || exit 1;

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

stage=1
stop_stage=3
nproc=64

log "$0 $*"
. utils/parse_options.sh

prefix=ReazonSpeech
output_dir=data/ReazonSpeech
splits="all"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 local/prepare_reazonspeech.py \
        --data_dir ${REAZONSPEECH} \
        --prefix ${prefix} \
        --output_dir ${output_dir} \
        --splits ${splits} \
        --nproc ${nproc} || exit 1;
fi

utt_extra_files="text.prev text.ctc"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" data/ReazonSpeech/${splits}
    utils/subset_data_dir_tr_cv.sh --cv-spk-percent 0.1 data/ReazonSpeech/${splits} \
        data/ReazonSpeech/train data/ReazonSpeech/test_valid
    utils/subset_data_dir_tr_cv.sh --cv-spk-percent 50  data/ReazonSpeech/test_valid \
        data/ReazonSpeech/valid data/ReazonSpeech/test
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    for x in train valid test; do
        # NOTE(yifan): extra text files must be sorted and unique
        for f in ${utt_extra_files}; do
            utils/filter_scp.pl -f 1 ${output_dir}/${x}/text ${output_dir}/${splits}/${f}\
              > ${output_dir}/${x}/${f}
            check_sorted ${output_dir}/${x}/${f}
        done
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${output_dir}/${x} || exit 1;
        utils/validate_data_dir.sh --no-feats --non-print ${output_dir}/${x} || exit 1;
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
