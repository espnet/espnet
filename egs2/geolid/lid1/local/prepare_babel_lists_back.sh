#!/usr/bin/env bash
# Script to prepare conf/lists files for BABEL dataset
# This script scans the BABEL corpus and creates the necessary list files

set -euo pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Configuration
dataset_path="/scratch/bbjs/shared/corpora/babel"
lists_dir="conf/lists"

# Language definitions - same as in prepare_babel.sh
langs=(
  "BABEL_101 yue 101-cantonese/train.FullLP.list 101-cantonese/dev.list"
  "BABEL_102 asm 102-assamese/train.FullLP.list 102-assamese/dev.list"
  "BABEL_103 ben 103-bengali/train.FullLP.list 103-bengali/dev.list"
  "BABEL_104 pus 104-pashto/training.list 104-pashto/dev.list"
  "BABEL_105 tur 105-turkish/train.FullLP.list 105-turkish/dev.list"
  "BABEL_106 tgl 106-tagalog/train.FullLP.list 106-tagalog/dev.list"
  "BABEL_107 vie 107-vietnamese/train.FullLP.list 107-vietnamese/dev.list"
  "BABEL_201 hat 201-haitian/train.FullLP.list 201-haitian/dev.list"
  "BABEL_202 swa 202-swahili/training.list 202-swahili/dev.list"
  "BABEL_203 lao 203-lao/train.FullLP.list 203-lao/dev.list"
  "BABEL_204 tam 204-tamil/train.FullLP.list 204-tamil/dev.list"
  "BABEL_205 kmr 205-kurmanji/training.list 205-kurmanji/dev.list"
  "BABEL_206 zul 206-zulu/train.FullLP.list 206-zulu/dev.list"
  "BABEL_207 tpi 207-tokpisin/training.list 207-tokpisin/dev.list"
  "BABEL_301 ceb 301-cebuano/training.list 301-cebuano/dev.list"
  "BABEL_302 kaz 302-kazakh/training.list 302-kazakh/dev.list"
  "BABEL_303 tel 303-telugu/training.list 303-telugu/dev.list"
  "BABEL_304 lit 304-lithuanian/training.list 304-lithuanian/dev.list"
  "BABEL_305 gug 305-guarani/training.list 305-guarani/dev.list"
  "BABEL_306 ibo 306-igbo/training.list 306-igbo/dev.list"
  "BABEL_307 amh 307-amharic/training.list 307-amharic/dev.list"
  "BABEL_401 khk 401-mongolian/training.list 401-mongolian/dev.list"
  "BABEL_402 jav 402-javanese/training.list 402-javanese/dev.list"
  "BABEL_403 luo 403-dholuo/training.list 403-dholuo/dev.list"
  "BABEL_404 kat 404-georgian/training.list 404-georgian/dev.list"
)

. utils/parse_options.sh

# Function to create list files for a given language and split
create_list_file() {
    local lang_dir="$1"
    local split_dir="$2"  # training or dev
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
    
    # Find all audio files and create list entries
    # BABEL typically uses .wav files, but also check for .flac and .sph
    local audio_files_found=0
    
    for ext in wav flac sph; do
        # Create a temporary file list
        local temp_files=$(find "$split_dir" -name "*.${ext}" -type f)
        
        if [ -n "$temp_files" ]; then
            local count=0
            while IFS= read -r audio_file; do
                # Extract filename without extension and path
                local basename=$(basename "$audio_file" ".${ext}")
                
                # BABEL format: Create utterance ID in format BABEL_XXX_filename
                local utt_id="${lang_code}_$(echo "$basename" | sed 's/[^a-zA-Z0-9_]/_/g')"
                
                # Add relative path from the base dataset directory
                local rel_path=$(realpath --relative-to="$lang_dir" "$audio_file")
                
                echo "$utt_id $rel_path" >> "$output_file"
                ((count++))
            done <<< "$temp_files"
            
            if [ $count -gt 0 ]; then
                audio_files_found=$count
                log "Found $audio_files_found ${ext} files in $split_dir"
                break
            fi
        fi
    done
    
    # If no audio files found, try to find transcription files and infer audio files
    if [ $audio_files_found -eq 0 ]; then
        log "No audio files found directly, checking for transcription files..."
        
        # Look for transcription files (common BABEL formats)
        for trans_file in "$split_dir"/*.txt "$split_dir"/transcription/*.txt; do
            if [ -f "$trans_file" ]; then
                # Extract file IDs from transcription files
                grep -E "^[A-Za-z0-9_]+\s" "$trans_file" 2>/dev/null | while read -r line; do
                    local file_id=$(echo "$line" | awk '{print $1}')
                    echo "$file_id" >> "$output_file"
                done 2>/dev/null || true
                
                local trans_entries=$(wc -l < "$output_file" 2>/dev/null || echo 0)
                if [ "$trans_entries" -gt 0 ]; then
                    log "Found $trans_entries entries from transcription file: $trans_file"
                    break
                fi
            fi
        done
    fi
    
    # Sort and deduplicate the list
    if [ -f "$output_file" ] && [ -s "$output_file" ]; then
        sort "$output_file" | uniq > "${output_file}.tmp"
        mv "${output_file}.tmp" "$output_file"
        local final_count=$(wc -l < "$output_file")
        log "Final list contains $final_count unique entries"
    else
        log "Warning: No entries found for $output_file"
        echo "# No audio files found in $split_dir" > "$output_file"
    fi
}

# Function to scan and create lists for a single language
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
    
    # Determine training directory path (BABEL audio files are in audio/ subdirectory)
    local train_dir=""
    for possible_train_dir in "$base_path/training/audio" "$base_path/training" "$base_path/train/audio" "$base_path/train"; do
        if [ -d "$possible_train_dir" ]; then
            train_dir="$possible_train_dir"
            break
        fi
    done
    
    # Determine dev directory path (BABEL audio files are in audio/ subdirectory)
    local dev_dir=""
    for possible_dev_dir in "$base_path/dev/audio" "$base_path/dev" "$base_path/development/audio" "$base_path/development"; do
        if [ -d "$possible_dev_dir" ]; then
            dev_dir="$possible_dev_dir"
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
main() {
    log "Starting BABEL lists preparation"
    log "Dataset path: $dataset_path"
    log "Output directory: $lists_dir"
    
    # Create lists directory
    mkdir -p "$lists_dir"
    
    # Check if dataset path exists
    if [ ! -d "$dataset_path" ]; then
        log "Error: Dataset path does not exist: $dataset_path"
        log "Please download and extract the BABEL corpus to: $dataset_path"
        log ""
        log "Expected structure:"
        log "$dataset_path/"
        log "├── BABEL_101/"
        log "│   └── conversational/"
        log "│       ├── training/"
        log "│       └── dev/"
        log "├── BABEL_102/"
        log "│   └── conversational/"
        log "│       ├── training/"
        log "│       └── dev/"
        log "└── ..."
        exit 1
    fi
    
    # Process each language
    local total_langs=${#langs[@]}
    local processed=0
    
    for entry in "${langs[@]}"; do
        read -r lang iso3 train_list dev_list <<< "$entry"
        
        ((processed++))
        log "[$processed/$total_langs] Processing $lang..."
        
        process_language "$lang" "$iso3" "$train_list" "$dev_list"
    done
    
    log "BABEL lists preparation completed!"
    log "Generated lists are available in: $lists_dir"
    log ""
    log "You can now run: bash local/prepare_babel.sh"
}

# Run main function
main "$@"
