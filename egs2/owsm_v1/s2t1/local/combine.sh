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

train_sets="dump/raw/AISHELL-1/train dump/raw/CoVoST2/train dump/raw/GigaSpeech/XL dump/raw/LibriSpeech/train-clean-100 dump/raw/LibriSpeech/train-clean-360 dump/raw/LibriSpeech/train-other-500 dump/raw/MuST-C_v1.2/train dump/raw/MuST-C_v2/train dump/raw/MuST-C_v3/train dump/raw/SPGISpeech/train dump/raw/TEDLIUM3/train"
dev_sets="dump/raw/AISHELL-1/dev dump/raw/CoVoST2/dev dump/raw/GigaSpeech/DEV dump/raw/LibriSpeech/dev-clean dump/raw/LibriSpeech/dev-other dump/raw/MuST-C_v1.2/dev dump/raw/MuST-C_v2/dev dump/raw/MuST-C_v3/dev dump/raw/SPGISpeech/val dump/raw/TEDLIUM3/dev"

train_out=dump/raw/train
dev_out=dump/raw/dev

mkdir -p ${train_out}
mkdir -p ${dev_out}

# Generate nlsyms
python3 local/generate_nlsyms.py

utt_extra_files="text.prev text.ctc"

# Combine dev
utils/combine_data.sh --skip_fix true --extra-files "${utt_extra_files}" ${dev_out} ${dev_sets} || exit 1;
# NOTE(yifan): extra text files must be sorted and unique
for f in ${utt_extra_files}; do
    check_sorted ${dev_out}/${f}
done
utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${dev_out} || exit 1;
utils/validate_data_dir.sh --no-feats --non-print ${dev_out} || exit 1;

# Combine train
utils/combine_data.sh --skip_fix true --extra-files "${utt_extra_files}" ${train_out} ${train_sets} || exit 1;
# NOTE(yifan): extra text files must be sorted and unique
for f in ${utt_extra_files}; do
    check_sorted ${train_out}/${f}
done
utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${train_out} || exit 1;
utils/validate_data_dir.sh --no-feats --non-print ${train_out} || exit 1;

log "Successfully finished. [elapsed=${SECONDS}s]"
