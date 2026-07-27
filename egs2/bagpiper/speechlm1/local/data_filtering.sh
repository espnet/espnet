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
SECONDS=0


stage=1
stop_stage=100000
nj=64

# Input/output directories
text_dir=""
audio_dir=""
output_dir=""

# Stage 1: Basic filtering parameters
min_caption_tokens=200
max_caption_tokens=800
min_audio_duration=0.0
max_audio_duration=30.0

# Repetition filtering parameters
max_char_repetition=5
max_word_repetition=3
max_ngram_repetition_ratio=0.1
ngram_n=5

# Lexical quality filtering parameters
min_unique_word_ratio=0.3
min_avg_word_length=2.0
max_avg_word_length=15.0
max_uppercase_ratio=0.5

log "$0 $*"
. utils/parse_options.sh

if [ -z "${text_dir}" ]; then
    log "Usage: $0 --text_dir <dir> --output_dir <dir> [--audio_dir <dir>]"
    log "  --stage              # Start stage (default: 1)"
    log "  --stop_stage         # Stop stage (default: 100000)"
    log "  --nj                 # Number of parallel workers (default: 64)"
    log "  --text_dir           # Input text directory"
    log "  --audio_dir          # Input audio directory (optional)"
    log "  --output_dir         # Output directory for filtered results"
    log "  --min_caption_tokens # Min caption tokens (default: 200)"
    log "  --max_caption_tokens # Max caption tokens (default: 800)"
    log "  --min_audio_duration # Min audio duration in seconds (default: 0.0)"
    log "  --max_audio_duration # Max audio duration in seconds (default: 30.0)"
    log "  --max_char_repetition      # Max consecutive repeated chars (default: 5)"
    log "  --max_word_repetition      # Max consecutive repeated words (default: 3)"
    log "  --max_ngram_repetition_ratio # Max n-gram repetition ratio (default: 0.1)"
    log "  --ngram_n            # N-gram size for repetition check (default: 5)"
    log "  --min_unique_word_ratio    # Min unique word ratio (default: 0.3)"
    log "  --min_avg_word_length      # Min avg word length (default: 2.0)"
    log "  --max_avg_word_length      # Max avg word length (default: 15.0)"
    log "  --max_uppercase_ratio      # Max uppercase ratio (default: 0.5)"
    exit 1
fi

if [ -z "${output_dir}" ]; then
    log "Error: --output_dir is required"
    exit 1
fi

mkdir -p "${output_dir}"


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Basic filtering (tokens, duration, repetition, lexical)"
    log "  Text directory: ${text_dir}"
    log "  Caption tokens: ${min_caption_tokens} - ${max_caption_tokens}"
    log "  Audio duration: ${min_audio_duration}s - ${max_audio_duration}s"
    log "  Max char repetition: ${max_char_repetition}"
    log "  Max word repetition: ${max_word_repetition}"
    log "  Max ${ngram_n}-gram repetition ratio: ${max_ngram_repetition_ratio}"
    log "  Min unique word ratio: ${min_unique_word_ratio}"
    log "  Avg word length: ${min_avg_word_length} - ${max_avg_word_length}"
    log "  Max uppercase ratio: ${max_uppercase_ratio}"

    python3 local/filter_stage1.py \
        --text_dir "${text_dir}" \
        --output_file "${output_dir}/stage1_survived_ids.txt" \
        --min_caption_tokens ${min_caption_tokens} \
        --max_caption_tokens ${max_caption_tokens} \
        --min_audio_duration ${min_audio_duration} \
        --max_audio_duration ${max_audio_duration} \
        --max_char_repetition ${max_char_repetition} \
        --max_word_repetition ${max_word_repetition} \
        --max_ngram_repetition_ratio ${max_ngram_repetition_ratio} \
        --ngram_n ${ngram_n} \
        --min_unique_word_ratio ${min_unique_word_ratio} \
        --min_avg_word_length ${min_avg_word_length} \
        --max_avg_word_length ${max_avg_word_length} \
        --max_uppercase_ratio ${max_uppercase_ratio} \
        --num_workers ${nj}

    # Count results
    total_survived=$(wc -l < "${output_dir}/stage1_survived_ids.txt")
    log "Stage 1 completed: ${total_survived} samples survived"
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
