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

help_message=$(cat << EOF
Usage: $0 [st-aeds-root] [--force]

Download and extract OpenSLR SLR45 / ST-AEDS.

Arguments:
    st-aeds-root: target directory. Defaults to ST_AEDS or downloads.

Options:
    --force: re-run extraction even when the completion marker exists.
EOF
)

root=${ST_AEDS:-downloads}
force=false
root_seen=false

while [ $# -gt 0 ]; do
    case "$1" in
        --force)
            force=true
            shift
            ;;
        -h|--help)
            echo "${help_message}"
            exit 0
            ;;
        -*)
            log "Error: unknown option $1"
            echo "${help_message}"
            exit 2
            ;;
        *)
            if "${root_seen}"; then
                log "Error: only one st-aeds-root argument is allowed."
                echo "${help_message}"
                exit 2
            fi
            root=$1
            root_seen=true
            shift
            ;;
    esac
done

archive_name=ST-AEDS-20180100_1-OS.tgz
archive_url=https://www.openslr.org/resources/45/${archive_name}
download_dir=${root}/downloads
archive_path=${download_dir}/${archive_name}
complete_marker=${root}/.complete

mkdir -p "${download_dir}"

if [ -f "${complete_marker}" ] && ! "${force}"; then
    log "${root} already has ${complete_marker}; skipping download/extraction."
    exit 0
fi

if [ -f "${archive_path}" ]; then
    log "Archive exists: ${archive_path}"
else
    log "Downloading ${archive_url} to ${archive_path}"
    if command -v wget >/dev/null 2>&1; then
        wget -O "${archive_path}" "${archive_url}"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o "${archive_path}" "${archive_url}"
    else
        log "Error: neither wget nor curl is available."
        exit 1
    fi
fi

if [ ! -s "${archive_path}" ]; then
    log "Error: archive is missing or empty: ${archive_path}"
    exit 1
fi

log "Extracting ${archive_path} into ${root}"
tar -xzf "${archive_path}" -C "${root}"

{
    echo "archive=${archive_name}"
    echo "url=${archive_url}"
    echo "completed_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} > "${complete_marker}"

log "Successfully downloaded and extracted ST-AEDS under ${root}"
