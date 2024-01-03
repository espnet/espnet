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

# Hint: You should link the train_v2 and dev_v2 to your dump/raw directory
# before running this script. You should also revise the language-IDs of v2
# data using local/filter_lang_id.py
train_sets="dump/raw/aidatatang/train_whisper \
            dump/raw/ami/ihm_train_whisper \
            dump/raw/CommonVoice/train \
            dump/raw/swbd/train_nodup_whisper \
            dump/raw/swbd/train_fisher_whisper \
            dump/raw/fisher_callhome/train_whisper \
            dump/raw/FLEURS/train \
            dump/raw/ksponspeech/train_whisper \
            dump/raw/magicdata/train_whisper \
            dump/raw/ReazonSpeech/train \
            dump/raw/ru_open_stt/train_whisper \
            dump/raw/vctk/tr_no_dev_whisper \
            dump/raw/VoxPopuli/train \
            dump/raw/voxforge/tr \
            dump/raw/babel/train \
            dump/raw/openslr/train \
            dump/raw/train_v2 \
            "
dev_sets="dump/raw/aidatatang/dev_whisper \
            dump/raw/ami/ihm_dev_whisper \
            dump/raw/CommonVoice/dev \
            dump/raw/swbd/train_dev_whisper \
            dump/raw/fisher_callhome/dev_whisper \
            dump/raw/FLEURS/valid \
            dump/raw/ksponspeech/dev_whisper \
            dump/raw/magicdata/dev_whisper \
            dump/raw/ReazonSpeech/valid \
            dump/raw/ru_open_stt/dev_whisper \
            dump/raw/vctk/dev_whisper \
            dump/raw/VoxPopuli/dev \
            dump/raw/voxforge/dt \
            dump/raw/babel/dev \
            dump/raw/openslr/dev \
            dump/raw/dev_v2 \
            "

train_out=dump/raw/train_v3
dev_out=dump/raw/dev_v3

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
