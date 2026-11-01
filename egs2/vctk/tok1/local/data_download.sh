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
url=https://datashare.ed.ac.uk/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/download

if [ -d "${corpus_dir}/wav48_silence_trimmed" ] && [ -d "${corpus_dir}/txt" ]; then
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
echo "Downloading VCTK 0.92 from the University of Edinburgh DataShare."
wget --continue --output-document="${archive}" "${url}"

echo "Extracting ${archive}."
unzip -q "${archive}" -d "${download_root}"
if [ ! -d "${corpus_dir}/wav48_silence_trimmed" ] \
        || [ ! -d "${corpus_dir}/txt" ]; then
    echo "Unexpected VCTK archive layout under ${corpus_dir}" >&2
    exit 1
fi

rm "${archive}"
echo "Successfully downloaded VCTK 0.92 to ${corpus_dir}."
