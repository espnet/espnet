#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# ============================================================================
# HEROICO (LDC2006S37) Spanish ASR recipe.
#
# Corpus: Spanish read + answer speech (heroico) plus the USMA prompt set.
# Pipeline: ESPnet2 asr.sh (Conformer encoder / Transformer decoder, hybrid
#           CTC/attention), character-level tokenization.
# ============================================================================

# ----- Corpus location / download (Stage -1) -------------------------------
# Path to the extracted LDC2006S37 corpus root (the dir that contains "data/").
# Edit this value (or export heroico_root before calling) to point at your copy.
# Note: extra CLI args are forwarded to asr.sh (e.g. --stage/--stop_stage/--ngpu).
heroico_root="${heroico_root:-./LDC2006S37}"
# OpenSLR mirror of the HEROICO corpus.
data_url=https://openslr.trmal.net/resources/39/LDC2006S37.tar.gz

# ----- Data sets -----------------------------------------------------------
train_set=train
valid_set=dev
test_sets="dev test"

# ----- Configs -------------------------------------------------------------
asr_config=conf/train_asr_conformer.yaml
inference_config=conf/decode_asr_transformer.yaml

lm_config=conf/train_lm_transformer.yaml
use_lm=false
use_wordlm=false

# Speed perturbation is disabled by default. asr.sh applies it in stage 2 and
# then renames the train set to "${train_set}_sp"; leaving it empty keeps the
# train set name as "train" so the recipe can also be started directly from
# stage 3 (see setup_and_train.sh). To enable it, set the factors below AND run
# from stage 1 so stage 2 actually builds data/train_sp.
speed_perturb_factors=

# ============================================================================
# Stage -1: download + extract the corpus if it is not already present.
# (asr.sh itself starts at stage 1, so we handle the raw download here.)
# ============================================================================
if [ ! -d "${heroico_root}/data" ]; then
    log "Stage -1: HEROICO not found at '${heroico_root}'. Downloading from ${data_url}"
    parent_dir="$(dirname "${heroico_root}")"
    mkdir -p "${parent_dir}"
    tarball="${parent_dir}/LDC2006S37.tar.gz"
    if [ ! -f "${tarball}" ]; then
        if command -v wget >/dev/null 2>&1; then
            wget --continue --tries=3 -O "${tarball}" "${data_url}"
        elif command -v curl >/dev/null 2>&1; then
            curl -L --retry 3 -o "${tarball}" "${data_url}"
        else
            log "Error: neither wget nor curl is available to download the corpus."
            exit 1
        fi
    fi
    log "Stage -1: Extracting ${tarball} -> ${parent_dir}"
    tar -xzf "${tarball}" -C "${parent_dir}"
else
    log "Stage -1: HEROICO already present at '${heroico_root}', skipping download."
fi

# ============================================================================
# Stage 1+ : standard ESPnet2 ASR pipeline.
#   --local_data_opts is forwarded to local/data.sh at asr.sh stage 1, i.e.
#       local/data.sh ${heroico_root}
#   which writes data/{train,dev,test}/{wav.scp,text,utt2spk,spk2utt}.
# ============================================================================
./asr.sh \
    --nj 16 \
    --inference_nj 16 \
    --ngpu 1 \
    --lang es \
    --local_data_opts "${heroico_root}" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --token_type char \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" "$@"
