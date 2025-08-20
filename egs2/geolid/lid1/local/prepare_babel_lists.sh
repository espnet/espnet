#!/usr/bin/env bash
# Minimal BABEL lists preparation script

set -euo pipefail

log() {
    echo "$(date): $*"
}

# Configuration
dataset_path="/scratch/bbjs/shared/corpora/babel"
lists_dir="conf/lists"

. utils/parse_options.sh

log "Starting minimal BABEL lists preparation"
mkdir -p "$lists_dir"

# Language definitions - full set (all 25 BABEL languages)
langs=(
    "BABEL_101:yue:101-cantonese:train.FullLP.list:dev.list"
    "BABEL_102:asm:102-assamese:train.FullLP.list:dev.list"
    "BABEL_103:ben:103-bengali:train.FullLP.list:dev.list"
    "BABEL_104:pus:104-pashto:training.list:dev.list"
    "BABEL_105:tur:105-turkish:train.FullLP.list:dev.list"
    "BABEL_106:tgl:106-tagalog:train.FullLP.list:dev.list"
    "BABEL_107:vie:107-vietnamese:train.FullLP.list:dev.list"
    "BABEL_201:hat:201-haitian:train.FullLP.list:dev.list"
    "BABEL_202:swa:202-swahili:training.list:dev.list"
    "BABEL_203:lao:203-lao:train.FullLP.list:dev.list"
    "BABEL_204:tam:204-tamil:train.FullLP.list:dev.list"
    "BABEL_205:kmr:205-kurmanji:training.list:dev.list"
    "BABEL_206:zul:206-zulu:train.FullLP.list:dev.list"
    "BABEL_207:tpi:207-tokpisin:training.list:dev.list"
    "BABEL_301:ceb:301-cebuano:training.list:dev.list"
    "BABEL_302:kaz:302-kazakh:training.list:dev.list"
    "BABEL_303:tel:303-telugu:training.list:dev.list"
    "BABEL_304:lit:304-lithuanian:training.list:dev.list"
    "BABEL_305:gug:305-guarani:training.list:dev.list"
    "BABEL_306:ibo:306-igbo:training.list:dev.list"
    "BABEL_307:amh:307-amharic:training.list:dev.list"
    "BABEL_401:khk:401-mongolian:training.list:dev.list"
    "BABEL_402:jav:402-javanese:training.list:dev.list"
    "BABEL_403:luo:403-dholuo:training.list:dev.list"
    "BABEL_404:kat:404-georgian:training.list:dev.list"
)

# Function to process a single language
process_language() {
    local lang_code="$1"
    local iso3="$2"
    local lang_dir="$3"
    local train_list="$4"
    local dev_list="$5"

    local base_dir="/scratch/bbjs/shared/corpora/babel/${lang_code}/conversational"
    local train_audio_dir="$base_dir/training/audio"
    local dev_audio_dir="$base_dir/dev/audio"
    local train_output="conf/lists/$lang_dir/$train_list"
    local dev_output="conf/lists/$lang_dir/$dev_list"

    log "Processing $lang_code ($iso3)"

    # Check if base directory exists
    if [ ! -d "$base_dir" ]; then
        log "Warning: Language directory not found: $base_dir, skipping..."
        return 0
    fi

    # Create output directories
    mkdir -p "$(dirname "$train_output")"
    mkdir -p "$(dirname "$dev_output")"

    # Process training files
    if [ -d "$train_audio_dir" ]; then
        > "$train_output"
        for file in "$train_audio_dir"/*.sph; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                utt_id="${lang_code}_$(echo "$filename" | sed 's/\.sph$//' | sed 's/[^a-zA-Z0-9_]/_/g')"
                echo "$utt_id training/audio/$filename" >> "$train_output"
            fi
        done
        local train_count=$(wc -l < "$train_output" 2>/dev/null || echo 0)
        log "Training list: $train_count entries"
    else
        log "Training directory not found: $train_audio_dir"
        touch "$train_output"  # Create empty file
    fi

    # Process dev files
    if [ -d "$dev_audio_dir" ]; then
        > "$dev_output"
        for file in "$dev_audio_dir"/*.sph; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                utt_id="${lang_code}_$(echo "$filename" | sed 's/\.sph$//' | sed 's/[^a-zA-Z0-9_]/_/g')"
                echo "$utt_id dev/audio/$filename" >> "$dev_output"
            fi
        done
        local dev_count=$(wc -l < "$dev_output" 2>/dev/null || echo 0)
        log "Dev list: $dev_count entries"
    else
        log "Dev directory not found: $dev_audio_dir"
        touch "$dev_output"  # Create empty file
    fi
}

# Process each language
for entry in "${langs[@]}"; do
    IFS=':' read -r lang_code iso3 lang_dir train_list dev_list <<< "$entry"
    process_language "$lang_code" "$iso3" "$lang_dir" "$train_list" "$dev_list"
done

log "Done! Processed ${#langs[@]} languages"

# Create .filenames format for perl script compatibility
log "Creating .filenames format for perl script..."
for lang_dir in conf/lists/*/; do
    if [ -d "$lang_dir" ]; then
        for listfile in "$lang_dir"/*.list; do
            if [ -f "$listfile" ] && [ -s "$listfile" ]; then
                awk '{print $2}' "$listfile" | sed 's|[^/]*/audio/||' | sed 's|\.sph$||' > "${listfile}.filenames"
                entries=$(wc -l < "${listfile}.filenames" 2>/dev/null || echo 0)
                if [ $entries -gt 0 ]; then
                    log "Created $(basename "$listfile").filenames with $entries entries"
                fi
            fi
        done
    fi
done

log "All list files and filenames created successfully!"
