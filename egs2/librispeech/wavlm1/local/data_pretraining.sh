#!/usr/bin/env bash
# Build espnet data dirs for SSL pre-training from the corpus metadata.
#
# Layout under --db_root:
#     audio/<corpus>/<lang>/**/*.flac
#     metadata/<corpus>/clips[_<lang>].parquet
#
# The parquet describes every clip (rel_path, measured_duration_s, ...), and all
# audio is already 16 kHz mono FLAC -- the `save_asis` case in
# format_wav_scp.py, where stage 3 copies the source path into wav.scp unchanged
# and only computes utt2num_samples. Both come straight from the metadata, so
# this writes dump/raw/org/<dset> as well and stage 3 has nothing left to do.
# See local/build_from_metadata.py for why the duration rounding is harmless.
#
# Produces one dir per <corpus>_<lang>, plus a held-out dev set and a combined
# train set. Re-running rebuilds from whatever metadata is present.
#
# NOTE ON TEXT: SSL pre-training does not use transcripts -- the targets are the
# k-means pseudo-labels in text.km.<km_tag>. But stage 4 of wavlm.sh filters on
# text and utils/ validation expects it, so a placeholder token is written per
# utterance. Do not use these dirs for a supervised task.
set -e
set -u
set -o pipefail

log() { echo "[$(date '+%F %T')] $*"; }

db_root=/mnt/weka/data/tagger_data/pretraining
dst_prefix=pretrain
train_set=pretrain_train
valid_set=pretrain_dev
dev_per_dset=150          # utterances held out per corpus/language
audio_ext=flac
dumpdir=dump/raw/org
scandir=data/.audio_scan     # cached filesystem listing, one file per corpus/lang
fs=16000
nj=8
stage=0
stop_stage=3

. utils/parse_options.sh

[ -d "${db_root}/audio" ]    || { log "ERROR: ${db_root}/audio missing"; exit 1; }
[ -d "${db_root}/metadata" ] || { log "ERROR: ${db_root}/metadata missing"; exit 1; }

# ------------------------------------------------- corpus/language <-> parquet
JOBS=()   # "<corpus> <lang> <parquet>"
for c in "${db_root}"/audio/*/; do
    c=${c%/}; corpus=$(basename "$c")
    for l in "$c"/*/; do
        l=${l%/}; lang=$(basename "$l")
        for p in "${db_root}/metadata/${corpus}/clips_${lang}.parquet" \
                 "${db_root}/metadata/${corpus}/clips.parquet"; do
            [ -f "$p" ] && { JOBS+=("${corpus} ${lang} ${p}"); break; }
        done
    done
done
[ ${#JOBS[@]} -gt 0 ] || { log "ERROR: no corpus/language pairs found"; exit 1; }
log "found ${#JOBS[@]} corpus/language pairs"

dsets=()
for j in "${JOBS[@]}"; do
    set -- $j; dsets+=("${dst_prefix}_$1_$2")
done

# --------------------------------------------------------- stage 0: scan audio
# The parquet is authoritative for durations but not necessarily complete, so
# list what is actually on disk and reconcile in stage 1.
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: listing audio under ${db_root}/audio (nj=${#JOBS[@]})"
    mkdir -p "${scandir}"
    printf '%s\n' "${JOBS[@]}" | xargs -P "${#JOBS[@]}" -I{} bash -c '
        set -- $1
        out="'"${scandir}"'/$1_$2.list"
        [ -s "${out}" ] && exit 0
        find -L "'"${db_root}"'/audio/$1/$2" -type f -name "*.'"${audio_ext}"'" \
            -printf "audio/$1/$2/%P\n" | LC_ALL=C sort > "${out}.tmp"
        mv "${out}.tmp" "${out}"
    ' _ {}
    log "stage 0: $(cat "${scandir}"/*.list | wc -l) files on disk"
fi

# ------------------------------------------------------- stage 1: per-dset dirs
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: building ${#JOBS[@]} data dirs (nj=${nj})"
    printf '%s\n' "${JOBS[@]}" | xargs -P "${nj}" -I{} bash -c '
        set -- $1
        python3 local/build_from_metadata.py \
            --parquet "$3" --corpus "$1" --lang "$2" \
            --db-root "'"${db_root}"'" --fs '"${fs}"' --audio-ext '"${audio_ext}"' \
            --audio-scan "'"${scandir}"'/$1_$2.list" \
            --data-dir "data/'"${dst_prefix}"'_$1_$2" \
            --dump-dir "'"${dumpdir}"'/'"${dst_prefix}"'_$1_$2"
    ' _ {}

    for d in "${dsets[@]}"; do
        n=$(wc -l < "data/${d}/wav.scp")
        [ "${n}" -gt 0 ] || { log "ERROR: data/${d} is empty"; exit 1; }
        [ "${n}" -eq "$(wc -l < "${dumpdir}/${d}/utt2num_samples")" ] \
            || { log "ERROR: ${d}: wav.scp / utt2num_samples length mismatch"; exit 1; }
    done
    log "stage 1: done"
fi

# ------------------------------------------- stage 2: dev split + combined train
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: holding out ${dev_per_dset} utts/dset -> ${valid_set}, rest -> ${train_set}"
    for split in "${train_set}" "${valid_set}"; do
        rm -rf "data/${split}" "${dumpdir}/${split}"
        mkdir -p "data/${split}" "${dumpdir}/${split}"
    done

    tmp=$(mktemp -d "${PWD}/data/.pretrain_split.XXXXXX")
    trap 'rm -rf "${tmp}"' EXIT

    # Deterministic hold-out: an evenly spaced slice of each dset, so the dev
    # set covers every corpus and language rather than whatever sorts first.
    for d in "${dsets[@]}"; do
        n=$(wc -l < "data/${d}/wav.scp")
        step=$(( n / dev_per_dset )); [ "${step}" -lt 1 ] && step=1
        # awk stops itself: piping into `head` would SIGPIPE it, and under
        # `set -o pipefail` that kills the script.
        awk -v s="${step}" -v k="${dev_per_dset}" \
            'NR % s == 1 {print $1; if (++c >= k) exit}' "data/${d}/wav.scp"
    done | LC_ALL=C sort -u > "${tmp}/devids"
    log "  dev ids: $(wc -l < "${tmp}/devids")"

    # Every per-dset file is already C-sorted, so merge rather than re-sort.
    for f in wav.scp utt2spk spk2utt text utt2num_samples; do
        srcs=()
        for d in "${dsets[@]}"; do
            if [ "${f}" = utt2num_samples ]; then srcs+=("${dumpdir}/${d}/${f}")
            else srcs+=("data/${d}/${f}"); fi
        done
        LC_ALL=C sort -m -k1,1 "${srcs[@]}" > "${tmp}/all.${f}"
        LC_ALL=C join -j1    "${tmp}/all.${f}" "${tmp}/devids" > "${dumpdir}/${valid_set}/${f}"
        LC_ALL=C join -v1 -j1 "${tmp}/all.${f}" "${tmp}/devids" > "${dumpdir}/${train_set}/${f}"
        rm -f "${tmp}/all.${f}"
    done

    for split in "${train_set}" "${valid_set}"; do
        for f in wav.scp utt2spk spk2utt text; do
            cp "${dumpdir}/${split}/${f}" "data/${split}/${f}"
        done
        echo raw > "${dumpdir}/${split}/feats_type"
        log "  ${split}: $(wc -l < "${dumpdir}/${split}/wav.scp") utts"
    done
fi

# ------------------------------------------------------------ stage 3: report
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: summary"
    for d in "${dsets[@]}" "${valid_set}" "${train_set}"; do
        [ -f "${dumpdir}/${d}/utt2num_samples" ] || continue
        awk -v n="${d}" -v fs="${fs}" \
            '{c++; s+=$2} END {printf "  %-30s %10d utts  %9.1f h\n", n, c, s/fs/3600}' \
            "${dumpdir}/${d}/utt2num_samples"
    done
fi
log "data preparation done"
