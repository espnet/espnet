#!/usr/bin/env bash
# Set bash to 'debug' mode
set -euo pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=100
dataset_name="babel"
train_set="train_${dataset_name}_lang"
valid_set="dev_${dataset_name}_lang"
dataset_path="downloads/babel"

log "$0 $*"
. utils/parse_options.sh
. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

partitions="${train_set} ${valid_set}"

langs=(
  "BABEL_101 yue 101-cantonese/train.FullLP.list 101-cantonese/dev.list"  "BABEL_102 asm 102-assamese/train.FullLP.list 102-assamese/dev.list"  "BABEL_103 ben 103-bengali/train.FullLP.list 103-bengali/dev.list"  "BABEL_104 pus 104-pashto/training.list 104-pashto/dev.list"
  "BABEL_105 tur 105-turkish/train.FullLP.list 105-turkish/dev.list"  "BABEL_106 tgl 106-tagalog/train.FullLP.list 106-tagalog/dev.list"  "BABEL_107 vie 107-vietnamese/train.FullLP.list 107-vietnamese/dev.list"  "BABEL_201 hat 201-haitian/train.FullLP.list 201-haitian/dev.list"
  "BABEL_202 swa 202-swahili/training.list 202-swahili/dev.list"  "BABEL_203 lao 203-lao/train.FullLP.list 203-lao/dev.list"  "BABEL_204 tam 204-tamil/train.FullLP.list 204-tamil/dev.list"  "BABEL_205 kmr 205-kurmanji/training.list 205-kurmanji/dev.list"
  "BABEL_206 zul 206-zulu/train.FullLP.list 206-zulu/dev.list"  "BABEL_207 tpi 207-tokpisin/training.list 207-tokpisin/dev.list"  "BABEL_301 ceb 301-cebuano/training.list 301-cebuano/dev.list"  "BABEL_302 kaz 302-kazakh/training.list 302-kazakh/dev.list"
  "BABEL_303 tel 303-telugu/training.list 303-telugu/dev.list"  "BABEL_304 lit 304-lithuanian/training.list 304-lithuanian/dev.list"  "BABEL_305 gug 305-guarani/training.list 305-guarani/dev.list"  "BABEL_306 ibo 306-igbo/training.list 306-igbo/dev.list"
  "BABEL_307 amh 307-amharic/training.list 307-amharic/dev.list"  "BABEL_401 khk 401-mongolian/training.list 401-mongolian/dev.list"  "BABEL_402 jav 402-javanese/training.list 402-javanese/dev.list"  "BABEL_403 luo 403-dholuo/training.list 403-dholuo/dev.list"
  "BABEL_404 kat 404-georgian/training.list 404-georgian/dev.list"
)

# =======================
# Stage 1: (Optional) Download
# =======================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Download BABEL"
    if [ ! -d "${dataset_path}" ]; then
        log "Please manually download the Babel corpus from LDC:"
        log "    https://catalog.ldc.upenn.edu"
        log "Then unzip it and place it under downloads/babel"
        log "Note: Due to copyright restrictions, we cannot provide an automatic download script."
        exit 1
    fi
fi

# =======================
# Stage 2: Prepare list files
# =======================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Preparing BABEL list files"
        
    bash local/prepare_babel_lists.sh --dataset_path ${dataset_path} --lists_dir conf/lists
        
    if [ $? -ne 0 ]; then
        log "Error: Failed to generate list files"
        exit 1
    fi
    
    log "✅ List files preparation complete"
fi

# =======================
# Stage 3: Prepare Kaldi-format files
# =======================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Preparing Kaldi-format files per language"

    for entry in "${langs[@]}"; do
        read -r lang iso3 train_list dev_list <<< "$entry"
        base_path="${dataset_path}/${lang}/conversational"
        lexicon_path="${base_path}/reference_materials/lexicon.txt"

        for split in training dev; do
            outname="train_${lang}"
            [ "$split" == "dev" ] && outname="dev_${lang}"
            out_dir="data/${outname}_${iso3}"

            log "Processing $lang ($iso3) $split -> $out_dir"
            mkdir -p "$out_dir"
            if [ "$split" == "training" ]; then
                local/prepare_babel.pl \
                    --iso3 "$iso3" --vocab "$lexicon_path" --fragmentMarkers \-\*\~ \
                    --filelist "conf/lists/${train_list}.filenames" \
                    "${base_path}/${split}" "$out_dir" > "$out_dir/skipped_utts.log"
            elif [ "$split" == "dev" ]; then
                local/prepare_babel.pl \
                    --iso3 "$iso3" --vocab "$lexicon_path" --fragmentMarkers \-\*\~ \
                    --filelist "conf/lists/${dev_list}.filenames" \
                    "${base_path}/${split}" "$out_dir" > "$out_dir/skipped_utts.log"
            fi
        done

    done

    # Combine all languages
    log "Combining all languages into train/dev sets"
    for split in train dev; do
        out_dir="data/${split}_babel_lang"
        mkdir -p "$out_dir"
        touch ${out_dir}/{utt2lang,segments,wav.scp}

        for d in data/${split}_BABEL_*; do
            log "Merging $d into $out_dir"
            for f in utt2lang segments wav.scp; do
                [ -f "$d/$f" ] && cat "$d/$f" >> "$out_dir/$f"
            done
        done

        for f in utt2lang segments wav.scp; do
            sort "$out_dir/$f" -o "$out_dir/$f"
        done

        utils/utt2spk_to_spk2utt.pl "$out_dir/utt2lang" > "$out_dir/lang2utt"

        # Move intermediate files to .backup
        for d in data/${split}_BABEL_*; do
            log "Backing up $d to data/.backup/"
            mv "$d" data/.backup/
        done

    done

    for x in ${partitions}; do
        cp "data/${x}/lang2utt" "data/${x}/category2utt"
    done

    log "✅ Kaldi preparation and merging complete"
fi

# =======================
# Stage 4: Validate data directories
# =======================
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Validating Kaldi data directories"
    for x in ${partitions}; do
        mv data/${x}/utt2lang data/${x}/utt2spk
        mv data/${x}/lang2utt data/${x}/spk2utt
        utils/fix_data_dir.sh "data/${x}" || exit 1
        utils/validate_data_dir.sh --no-feats "data/${x}" --no-text
        mv data/${x}/utt2spk data/${x}/utt2lang
        mv data/${x}/spk2utt data/${x}/lang2utt
    done
    log "✅ Validation complete"
fi

log "Finished in $SECONDS seconds"
