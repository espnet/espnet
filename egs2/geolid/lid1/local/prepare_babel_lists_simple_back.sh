#!/usr/bin/env bash
# Simplified BABEL lists preparation script

set -euo pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Configuration
dataset_path="/scratch/bbjs/shared/corpora/babel"
lists_dir="conf/lists"

# Just test with a few languages first
langs=(
  "BABEL_101 yue 101-cantonese/train.FullLP.list 101-cantonese/dev.list"
  "BABEL_102 asm 102-assamese/train.FullLP.list 102-assamese/dev.list"
)

# Function to create list file for a given language and split
create_list_file() {
    local base_path="$1"
    local split_dir="$2"
    local output_file="$3"
    local lang_code="$4"
    
    log "Creating list file: $output_file"
    log "Scanning directory: $split_dir"
    
    # Create output directory if it doesn't exist
    mkdir -p "$(dirname "$output_file")"
    
    # Clear the output file
    > "$output_file"
    
    if [ ! -d "$split_dir" ]; then
        log "Warning: Directory $split_dir does not exist, skipping..."
        return 0
    fi
    
    # Find .sph files (BABEL format)
    local count=0
    while IFS= read -r audio_file; do
        if [ -n "$audio_file" ]; then
            local basename=$(basename "$audio_file" ".sph")
            local utt_id="${lang_code}_$(echo "$basename" | sed 's/[^a-zA-Z0-9_]/_/g')"
            local rel_path=$(realpath --relative-to="$base_path" "$audio_file")
            echo "$utt_id $rel_path" >> "$output_file"
            ((count++))
        fi
    done < <(find "$split_dir" -name "*.sph" -type f)
    
    log "Found $count .sph files in $split_dir"
    
    # Sort the output file
    if [ -f "$output_file" ] && [ -s "$output_file" ]; then
        sort "$output_file" -o "$output_file"
        log "List file created successfully: $output_file"
    fi
}

# Function to process a single language
process_language() {
    local lang="$1"
    local iso3="$2"
    local train_list="$3"
    local dev_list="$4"
    
    log "Processing language: $lang ($iso3)"
    
    local base_path="${dataset_path}/${lang}/conversational"
    
    # Check if the language directory exists
    if [ ! -d "$base_path" ]; then
        log "Warning: Language directory not found: $base_path"
        log "Skipping $lang..."
        return 0
    fi
    
    # Find training directory
    local train_dir=""
    for possible_dir in "$base_path/training/audio" "$base_path/training"; do
        if [ -d "$possible_dir" ]; then
            train_dir="$possible_dir"
            break
        fi
    done
    
    # Find dev directory
    local dev_dir=""
    for possible_dir in "$base_path/dev/audio" "$base_path/dev"; do
        if [ -d "$possible_dir" ]; then
            dev_dir="$possible_dir"
            break
        fi
    done
    
    # Create training list
    if [ -n "$train_dir" ]; then
        create_list_file "$base_path" "$train_dir" "${lists_dir}/${train_list}" "$lang"
    else
        log "Warning: No training directory found for $lang"
    fi
    
    # Create dev list
    if [ -n "$dev_dir" ]; then
        create_list_file "$base_path" "$dev_dir" "${lists_dir}/${dev_list}" "$lang"
    else
        log "Warning: No dev directory found for $lang"
    fi
}

# Main execution
log "Starting simplified BABEL lists preparation"
log "Dataset path: $dataset_path"
log "Output directory: $lists_dir"

# Create lists directory
mkdir -p "$lists_dir"

# Process each language
for entry in "${langs[@]}"; do
    read -r lang iso3 train_list dev_list <<< "$entry"
    process_language "$lang" "$iso3" "$train_list" "$dev_list"
done

log "BABEL lists preparation completed!"
