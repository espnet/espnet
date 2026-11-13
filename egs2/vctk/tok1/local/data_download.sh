#!/usr/bin/env bash

set -e
set -u
set -o pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <download-root>" >&2
    exit 2
fi

download_root=$1
corpus_dir=${download_root}/VCTK-Corpus-0.92
legacy_corpus_dir=${download_root}/VCTK-Corpus
archive=${download_root}/VCTK-Corpus-0.92.zip
complete_marker=${corpus_dir}/.complete
url=https://datashare.ed.ac.uk/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/download

if [ -f "${complete_marker}" ]; then
    echo "${corpus_dir} already exists; download is skipped."
    exit 0
fi
if [ -d "${legacy_corpus_dir}/wav48" ] && [ -d "${legacy_corpus_dir}/txt" ]; then
    echo "${legacy_corpus_dir} already exists; VCTK 0.92 download is skipped."
    exit 0
fi
# Also accept VCTK itself, rather than its parent, as the configured path.
if { [ -d "${download_root}/wav48_silence_trimmed" ] \
        || [ -d "${download_root}/wav48" ]; } && [ -d "${download_root}/txt" ]; then
    echo "${download_root} is an existing VCTK corpus; download is skipped."
    exit 0
fi

mkdir -p "${download_root}"
if [ -f "${archive}" ]; then
    if unzip -tq "${archive}"; then
        echo "${archive} exists and appears to be complete."
    else
        echo "Removing invalid archive ${archive}."
        rm "${archive}"
    fi
fi

if [ ! -f "${archive}" ]; then
    echo "Downloading VCTK 0.92 from the University of Edinburgh DataShare."
    if ! wget --output-document="${archive}" "${url}"; then
        echo "Failed to download ${url}" >&2
        exit 1
    fi
    if ! unzip -tq "${archive}"; then
        echo "Downloaded archive is invalid: ${archive}" >&2
        rm "${archive}"
        exit 1
    fi
fi

echo "Extracting ${archive}."
if ! unzip -qo "${archive}" -d "${download_root}"; then
    echo "Failed to extract ${archive}" >&2
    exit 1
fi
if [ ! -d "${corpus_dir}/wav48_silence_trimmed" ] \
        || [ ! -d "${corpus_dir}/txt" ]; then
    echo "Unexpected VCTK archive layout under ${corpus_dir}" >&2
    exit 1
fi

touch "${complete_marker}"
rm "${archive}"
echo "Successfully downloaded VCTK 0.92 to ${corpus_dir}."
