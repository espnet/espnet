#!/usr/bin/env bash
# local/data.sh — Speech Cleaner data preparation (Samsung PC)
#
# Required (export from db.sh):
#   DATASET_LIBRITTS_R  /DB/LibriTTS_R    24kHz
#   DATASET_EARS        /DB/ears          48kHz  p001-p107
#   DATASET_VCTK_DEMAND /DB/VCTK_DEMAND   48kHz  clean dirs only
#
# Outputs:
#   data/train_fp   LibriTTS-R train + EARS p001-p096 + VCTK clean train
#   data/dev_fp     LibriTTS-R dev  + EARS p097-p107 + VCTK clean test
#   data/train_voc  EARS p001-p096  + VCTK clean train  (48kHz only, no LibriTTS-R)
#   data/dev_voc    EARS p097-p107  + VCTK clean test   (48kHz only, no LibriTTS-R)
#   data/noise_pool symlinks from all NOISE_* dirs

set -euo pipefail
log() { echo "[data.sh $(date '+%H:%M:%S')] $*"; }

# ── Validate required variables ────────────────────────────────────────────
for _var in DATASET_LIBRITTS_R DATASET_EARS DATASET_VCTK_DEMAND; do
    _val="${!_var:-}"
    [ -n "${_val}" ] && [ -d "${_val}" ] || {
        log "ERROR: ${_var} not set or not a directory (value: '${_val:-<unset>}')"
        exit 1
    }
done

# ─────────────────────────────────────────────────────────────────────────
# _make_kaldi INPUT OUT
#   INPUT: file with 3 fields per line:  uttid  wavpath  speaker
#   Writes wav.scp, utt2spk, spk2utt into OUT/
# ─────────────────────────────────────────────────────────────────────────
_make_kaldi() {
    local input=$1 out=$2
    mkdir -p "${out}"
    awk '{print $1, $2}' "${input}" | sort -u > "${out}/wav.scp"
    awk '{print $1, $3}' "${input}" | sort -u > "${out}/utt2spk"
    awk '{
        spk=$2; utt=$1
        spk2utt[spk] = (spk in spk2utt) ? spk2utt[spk]" "utt : utt
    } END {
        for (s in spk2utt) print s, spk2utt[s]
    }' "${out}/utt2spk" | sort > "${out}/spk2utt"
    log "  $(wc -l < "${out}/wav.scp") utts → ${out}"
}

# ─────────────────────────────────────────────────────────────────────────
# _collect_libritts_r SUBSET [SUBSET ...]
#   Emits: "ltr_{spk}_{stem}  /path/file  {spk}"
#   Speaker = two levels up from file (reader directory)
# ─────────────────────────────────────────────────────────────────────────
_collect_libritts_r() {
    for sub in "$@"; do
        local d="${DATASET_LIBRITTS_R}/${sub}"
        [ -d "${d}" ] || { log "  SKIP LibriTTS-R/${sub}"; continue; }
        find "${d}" \( -name "*.wav" -o -name "*.flac" \) | sort
    done | awk '{
        n=split($0,a,"/"); fname=a[n]; spk=a[n-2]
        sub(/\.(wav|flac)$/, "", fname)
        gsub(/[^A-Za-z0-9_-]/, "_", fname)
        gsub(/[^A-Za-z0-9_-]/, "_", spk)
        print "ltr_"spk"_"fname, $0, spk
    }'
}

# ─────────────────────────────────────────────────────────────────────────
# _collect_ears train|dev
#   train → p001-p096  (90%)
#   dev   → p097-p107  (10%)
#   Emits: "ears_{spk}_{stem}  /path/file  {spk}"
# ─────────────────────────────────────────────────────────────────────────
_collect_ears() {
    local split=$1
    find "${DATASET_EARS}" -mindepth 1 -maxdepth 1 -type d -name 'p[0-9][0-9][0-9]' | \
    sort | while IFS= read -r spk_dir; do
        spk=$(basename "${spk_dir}")
        # awk converts "097" → 97 (no octal), safe for leading-zero speaker nums
        num=$(echo "${spk}" | awk '{n=$0; sub(/^p/,"",n); print n+0}')
        if   { [ "${split}" = "train" ] && [ "${num}" -le 96 ]; } \
          || { [ "${split}" = "dev"   ] && [ "${num}" -ge 97 ]; }; then
            find "${spk_dir}" \( -name "*.wav" -o -name "*.flac" \) | sort
        fi
    done | awk '{
        n=split($0,a,"/"); fname=a[n]; spk=a[n-1]
        sub(/\.(wav|flac)$/, "", fname)
        gsub(/[^A-Za-z0-9_-]/, "_", fname)
        print "ears_"spk"_"fname, $0, spk
    }'
}

# ─────────────────────────────────────────────────────────────────────────
# _collect_vctk_clean SUBDIR [SUBDIR ...]
#   Only clean_* subdirs accepted; noisy_* are rejected with error.
#   Speaker extracted from filename prefix: p225_001.wav → p225
#   Works for both flat and speaker-subdir layouts.
#   Emits: "vctk_{spk}_{stem}  /path/file  {spk}"
# ─────────────────────────────────────────────────────────────────────────
_collect_vctk_clean() {
    for sub in "$@"; do
        case "${sub}" in
            noisy_*|mix_*)
                log "ERROR: noisy/mix dir '${sub}' must not be included"; exit 1 ;;
        esac
        local d="${DATASET_VCTK_DEMAND}/${sub}"
        [ -d "${d}" ] || { log "  SKIP VCTK_DEMAND/${sub}"; continue; }
        find "${d}" \( -name "*.wav" -o -name "*.flac" \) | sort
    done | awk '{
        n=split($0,a,"/"); fname=a[n]
        sub(/\.(wav|flac)$/, "", fname)
        # Speaker = first "_"-delimited token of filename: p225_001 → p225
        split(fname, sp, "_"); spk=sp[1]
        gsub(/[^A-Za-z0-9_-]/, "_", fname)
        print "vctk_"spk"_"fname, $0, spk
    }'
}

# ─────────────────────────────────────────────────────────────────────────
# Collect
# ─────────────────────────────────────────────────────────────────────────
tmp=$(mktemp -d); trap "rm -rf ${tmp}" EXIT

log "--- LibriTTS-R train (train-clean-100/360, train-other-500) ---"
_collect_libritts_r train-clean-100 train-clean-360 train-other-500 \
    > "${tmp}/ltr_train"
log "  $(wc -l < "${tmp}/ltr_train") utts"

log "--- LibriTTS-R dev (dev-clean, dev-other) ---"
_collect_libritts_r dev-clean dev-other > "${tmp}/ltr_dev"
log "  $(wc -l < "${tmp}/ltr_dev") utts"

log "--- LibriTTS-R test ---"
_collect_libritts_r test-clean > "${tmp}/ltr_test_clean"
_collect_libritts_r test-other > "${tmp}/ltr_test_other"

log "--- EARS train (p001-p096) ---"
_collect_ears train > "${tmp}/ears_train"
log "  $(wc -l < "${tmp}/ears_train") utts"

log "--- EARS dev (p097-p107) ---"
_collect_ears dev > "${tmp}/ears_dev"
log "  $(wc -l < "${tmp}/ears_dev") utts"

log "--- VCTK_DEMAND clean train (28spk + 56spk) ---"
_collect_vctk_clean clean_trainset_28spk_wav clean_trainset_56spk_wav \
    > "${tmp}/vctk_train"
log "  $(wc -l < "${tmp}/vctk_train") utts"

log "--- VCTK_DEMAND clean dev (testset) ---"
_collect_vctk_clean clean_testset_wav > "${tmp}/vctk_dev"
log "  $(wc -l < "${tmp}/vctk_dev") utts"

# Sanity check: none should be empty
for _f in ltr_train ltr_dev ears_train ears_dev vctk_train vctk_dev; do
    [ -s "${tmp}/${_f}" ] || { log "ERROR: ${_f} is empty — check paths"; exit 1; }
done

# ─────────────────────────────────────────────────────────────────────────
# Assemble Kaldi dirs
# ─────────────────────────────────────────────────────────────────────────
log "Building data/train_fp  (LibriTTS-R + EARS train + VCTK clean train) ..."
cat "${tmp}/ltr_train" "${tmp}/ears_train" "${tmp}/vctk_train" \
    | sort -u > "${tmp}/fp_train"
_make_kaldi "${tmp}/fp_train" data/train_fp

log "Building data/dev_fp  (LibriTTS-R dev + EARS dev + VCTK clean dev) ..."
cat "${tmp}/ltr_dev" "${tmp}/ears_dev" "${tmp}/vctk_dev" \
    | sort -u > "${tmp}/fp_dev"
_make_kaldi "${tmp}/fp_dev" data/dev_fp

_make_kaldi "${tmp}/ltr_test_clean" data/test-clean
_make_kaldi "${tmp}/ltr_test_other" data/test-other

log "Building data/train_voc  (EARS train + VCTK clean train, 48kHz only) ..."
cat "${tmp}/ears_train" "${tmp}/vctk_train" \
    | sort -u > "${tmp}/voc_train"
_make_kaldi "${tmp}/voc_train" data/train_voc

log "Building data/dev_voc  (EARS dev + VCTK clean dev, 48kHz only) ..."
cat "${tmp}/ears_dev" "${tmp}/vctk_dev" \
    | sort -u > "${tmp}/voc_dev"
_make_kaldi "${tmp}/voc_dev" data/dev_voc

# ─────────────────────────────────────────────────────────────────────────
# Noise pool
# ─────────────────────────────────────────────────────────────────────────
log "Building data/noise_pool ..."
mkdir -p data/noise_pool
_noise_found=0
for _var in $(compgen -v | grep '^NOISE_' | sort); do
    _dir="${!_var:-}"
    [ -z "${_dir}" ] && continue
    [ -d "${_dir}" ] || { log "  SKIP ${_var}: not found"; continue; }
    log "  ${_var}: ${_dir}"
    find "${_dir}" \( -name "*.wav" -o -name "*.flac" \) | \
        while IFS= read -r f; do
            ln -sf "${f}" "data/noise_pool/${_var}_$(basename "${f}")" 2>/dev/null || true
        done
    _noise_found=1
done
[ "${_noise_found}" -eq 0 ] && log "WARNING: No NOISE_* variables set."
log "  $(ls data/noise_pool | wc -l) noise files"

# ─────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────
log "=== Done ==="
log "  train_fp : $(wc -l < data/train_fp/wav.scp) utts"
log "  dev_fp   : $(wc -l < data/dev_fp/wav.scp) utts"
log "  test-clean: $(wc -l < data/test-clean/wav.scp) utts"
log "  test-other: $(wc -l < data/test-other/wav.scp) utts"
log "  train_voc: $(wc -l < data/train_voc/wav.scp) utts"
log "  dev_voc  : $(wc -l < data/dev_voc/wav.scp) utts"
log "  noise_pool: $(ls data/noise_pool | wc -l) files"