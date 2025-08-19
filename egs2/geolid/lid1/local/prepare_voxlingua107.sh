#!/usr/bin/env bash
# Create data directories if they do not exist
log() {
    local fname=${BASH_SOURCE[1]##*/}
        echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p data
mkdir -p data/train_voxlingua107_lang
mkdir -p data/dev_voxlingua107_lang

# generate wav.scp
log "Generating wav.scp for train and dev sets"
find /scratch/bbjs/shared/corpora/voxlingua107 -type f -name "*.wav" | awk -F '/' '{print $NF, $0}' > data/train_voxlingua107_lang/wav.scp
find /scratch/bbjs/shared/corpora/voxlingua107/dev -type f -name "*.wav" | awk -F '/' '{print $NF, $0}' > data/dev_voxlingua107_lang/wav.scp

log "Clear utterances in train also in dev set"
python local/prepare_voxlingua107.py --func_name clear_dev_from_train

log "Generating utt2spk for train and dev sets"
python local/prepare_voxlingua107.py --func_name gen_utt2spk

log "Converting utt2spk to spk2utt for train and dev sets"
./utils/utt2spk_to_spk2utt.pl data/train_voxlingua107_lang/utt2spk > data/train_voxlingua107_lang/spk2utt
./utils/utt2spk_to_spk2utt.pl data/dev_voxlingua107_lang/utt2spk > data/dev_voxlingua107_lang/spk2utt

log "Copying spk2utt to category2utt for train and dev sets"
cp data/train_voxlingua107_lang/spk2utt data/train_voxlingua107_lang/category2utt
cp data/dev_voxlingua107_lang/spk2utt data/dev_voxlingua107_lang/category2utt

utils/fix_data_dir.sh data/train_voxlingua107_lang || exit 1;
utils/fix_data_dir.sh data/dev_voxlingua107_lang || exit 1;
utils/validate_data_dir.sh --no-feats --non-print --no-text data/train_voxlingua107_lang || exit 1;
utils/validate_data_dir.sh --no-feats --non-print --no-text data/dev_voxlingua107_lang || exit 1;
