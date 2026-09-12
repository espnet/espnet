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
stop_stage=100

# Transcription tier of the TextGrid files:
#   utt.ortho.  orthographic  (standard spelling, e.g. 제 이름은)
#   utt.prono.  pronounced    (as actually spoken,  e.g. 제 이르믄)
tier=utt.ortho.
min_duration=0.2
max_duration=20.0
# Where the corpus is unpacked to when only the distributed archives are found.
unpack_dir=downloads

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${SEOUL_CORPUS}" ]; then
    log "Error: \$SEOUL_CORPUS is not set. Set it in db.sh to the directory that"
    log "       holds the Seoul Corpus, i.e. either the distributed archives"
    log "         sound.tgz  label.tgz"
    log "       or already unpacked 'sound/' and 'label/' subdirectories."
    exit 2
fi

# The 40 speakers come in 8 balanced groups of 5 (male/female x teens, twenties,
# thirties, forties).  Each group gives up exactly one speaker, to dev or to test,
# so that the three sets are speaker-disjoint and dev and test each cover all four
# age groups with two male and two female speakers.
#   dev : s10 f/teens  s15 m/twenties  s30 f/thirties  s35 m/forties
#   test: s05 m/teens  s20 f/twenties  s25 m/thirties  s40 f/forties
dev_speakers="s10 s15 s30 s35"
test_speakers="s05 s20 s25 s40"
train_speakers=$(for i in $(seq -w 1 40); do
    case " ${dev_speakers} ${test_speakers} " in
        *" s${i} "*) ;;
        *) echo -n "s${i} " ;;
    esac
done)

sound_dir=
label_dir=

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Locate / unpack the Seoul Corpus in ${SEOUL_CORPUS}"

    # unpack <name> <destination> <archive> ...
    # Accepts the .tgz shipped by the distributor (which wraps a .zip), a bare
    # .zip, or a directory that already holds the files.
    unpack() {
        local pattern=$1 dst=$2; shift 2
        if [ -n "$(find "${dst}" -maxdepth 1 -name "${pattern}" -print -quit 2>/dev/null)" ]; then
            log "  already unpacked: ${dst}"
            return
        fi
        local src
        for src in "$@"; do
            [ -e "${src}" ] || continue
            log "  unpacking ${src} -> ${dst}"
            mkdir -p "${dst}"
            case "${src}" in
                *.tgz|*.tar.gz)
                    local tmp="${dst}.tmp"
                    rm -rf "${tmp}"; mkdir -p "${tmp}"
                    tar xzf "${src}" -C "${tmp}"
                    # the archives contain a single zip file
                    find "${tmp}" -name '*.zip' -exec unzip -q -o {} -d "${dst}" \;
                    find "${tmp}" \( -name '*.flac' -o -name '*.TextGrid' \) \
                        -exec mv {} "${dst}" \;
                    rm -rf "${tmp}"
                    ;;
                *.zip)
                    unzip -q -o "${src}" -d "${dst}"
                    ;;
                *)
                    log "Error: do not know how to unpack ${src}"; exit 1
                    ;;
            esac
            # macOS resource forks that ship inside the zips
            rm -rf "${dst}/__MACOSX"
            find "${dst}" -name '._*' -delete
            return
        done
        log "Error: found none of [$*]; cannot obtain ${pattern} files."
        exit 1
    }

    if [ -n "$(find "${SEOUL_CORPUS}/sound" -maxdepth 1 -name '*.flac' -print -quit 2>/dev/null)" ]; then
        sound_dir="${SEOUL_CORPUS}/sound"
    elif [ -n "$(find "${SEOUL_CORPUS}" -maxdepth 1 -name '*.flac' -print -quit 2>/dev/null)" ]; then
        sound_dir="${SEOUL_CORPUS}"
    else
        sound_dir="${unpack_dir}/sound"
        unpack '*.flac' "${sound_dir}" \
            "${SEOUL_CORPUS}/sound.tgz" "${SEOUL_CORPUS}/sound-flac.zip"
    fi

    if [ -n "$(find "${SEOUL_CORPUS}/label" -maxdepth 1 -name '*.TextGrid' -print -quit 2>/dev/null)" ]; then
        label_dir="${SEOUL_CORPUS}/label"
    elif [ -n "$(find "${SEOUL_CORPUS}" -maxdepth 1 -name '*.TextGrid' -print -quit 2>/dev/null)" ]; then
        label_dir="${SEOUL_CORPUS}"
    else
        label_dir="${unpack_dir}/label"
        unpack '*.TextGrid' "${label_dir}" \
            "${SEOUL_CORPUS}/label.tgz" "${SEOUL_CORPUS}/label-TextGrid.zip"
    fi

    mkdir -p "${unpack_dir}"
    log "  sound: ${sound_dir} ($(find "${sound_dir}" -maxdepth 1 -name '*.flac' | wc -l) flac)"
    log "  label: ${label_dir} ($(find "${label_dir}" -maxdepth 1 -name '*.TextGrid' | wc -l) TextGrid)"
    echo "${sound_dir}" > "${unpack_dir}/.sound_dir"
    echo "${label_dir}" > "${unpack_dir}/.label_dir"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Build Kaldi data directories from the TextGrid labels"

    sound_dir=$(cat "${unpack_dir}/.sound_dir")
    label_dir=$(cat "${unpack_dir}/.label_dir")

    for part in train dev test; do
        eval "speakers=\${${part}_speakers}"

        # The duration filter is for training data only: the test set has to be
        # scored on every utterance the corpus annotates, or the reference text is
        # not the corpus's any more and the score stops being comparable.
        if [ "${part}" = test ]; then
            _min_duration=0
            _max_duration=100000
        else
            _min_duration="${min_duration}"
            _max_duration="${max_duration}"
        fi

        python3 local/prepare_data.py \
            --sound_dir "${sound_dir}" \
            --label_dir "${label_dir}" \
            --out_dir "data/${part}" \
            --speakers "${speakers}" \
            --tier "${tier}" \
            --min_duration "${_min_duration}" \
            --max_duration "${_max_duration}"

        utils/utt2spk_to_spk2utt.pl "data/${part}/utt2spk" > "data/${part}/spk2utt"
        utils/fix_data_dir.sh "data/${part}"
        utils/validate_data_dir.sh --no-feats "data/${part}"
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
