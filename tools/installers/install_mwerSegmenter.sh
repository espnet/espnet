#!/usr/bin/env bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

unames="$(uname -s)"
if [[ ! ${unames} =~ Linux && ! ${unames} =~ Darwin ]]; then
    echo "Warning: This script may not work with ${unames}. Exit with doing nothing"
    exit 0
fi

PRIMARY_URL="https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz"
BACKUP_URL="https://huggingface.co/espnet/ci_tools/resolve/main/mwerSegmenter.tar.gz"
ARCHIVE_NAME="mwerSegmenter.tar.gz"
ARCHIVE_DIR_PATTERN='^(\./)?mwerSegmenter/'

download_archive() {
    local url="$1"
    local output="$2"
    local tmp_output="${output}.tmp"

    rm -f "${tmp_output}"
    wget --no-check-certificate --tries=5 --waitretry=5 --retry-connrefused \
        --timeout=30 --read-timeout=30 -O "${tmp_output}" "${url}"
    mv "${tmp_output}" "${output}"
}

validate_archive() {
    local archive_path="$1"
    local archive_listing

    archive_listing="$(tar tf "${archive_path}")" || return 1
    printf '%s\n' "${archive_listing}" | awk -v pattern="${ARCHIVE_DIR_PATTERN}" '$0 ~ pattern {found=1; exit} END {exit !found}'
}

download_and_validate_archive() {
    local url="$1"
    local output="$2"

    download_archive "${url}" "${output}" || return 1
    validate_archive "${output}"
}

if [ -f "${ARCHIVE_NAME}" ] && validate_archive "${ARCHIVE_NAME}"; then
    echo "Using cached archive ${ARCHIVE_NAME}"
else
    rm -f "${ARCHIVE_NAME}"
    if ! download_and_validate_archive "${PRIMARY_URL}" "${ARCHIVE_NAME}"; then
        echo "Primary download failed, trying backup URL..."
        echo ""
        echo "=============================================================================="
        echo "LEGAL DISCLAIMER:"
        echo "The backup URL (HuggingFace mirror) is provided only for installation purposes."
        echo "The mwerSegmenter software is subject to the RWTH mwerSegmenter License."
        echo ""
        echo "This software is for NON-COMMERCIAL USE ONLY."
        echo ""
        echo "Key terms of the RWTH mwerSegmenter License:"
        echo "- Non-exclusive rights are granted for non-commercial use only."
        echo "- Any commercial use or distribution requires prior authorization by RWTH."
        echo "- The software is provided 'AS IS' with no warranty of any kind."
        echo "- This license is subject to German law."
        echo ""
        echo "For the full license text, please refer to the original RWTH mwerSegmenter"
        echo "distribution or contact RWTH directly."
        echo "=============================================================================="
        echo ""
        rm -f "${ARCHIVE_NAME}"
        if ! download_and_validate_archive "${BACKUP_URL}" "${ARCHIVE_NAME}"; then
            echo "Both primary and backup downloads failed"
            exit 1
        fi
    fi
fi

tar zxvf "${ARCHIVE_NAME}"
rm -f "${ARCHIVE_NAME}"

patch mwerSegmenter/hyp2sgm.py < installers/patch_mwerSegmenter/hyp2sgm.patch
patch mwerSegmenter/sgm2mref.py < installers/patch_mwerSegmenter/sgm2mref.patch
