#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <download_dir>" >&2
    exit 1
fi

download_dir=$1
corpus_name=ThorstenVoice-Dataset_2022.10
archive_name=${corpus_name}.zip
corpus_dir="${download_dir}/${corpus_name}"
archive="${download_dir}/${archive_name}"

url="https://zenodo.org/records/7265581/files/${archive_name}?download=1"
md5="c2c2cb0d8a2b3b240e140d9213cd39b8"

if [ -f "${corpus_dir}/metadata_train.csv" ] && [ -d "${corpus_dir}/wavs" ]; then
    echo "Thorsten dataset already exists. Skipped."
    exit 0
fi

mkdir -p "${download_dir}"

if [ ! -f "${archive}" ]; then
    echo "Downloading Thorsten-Voice 2022.10..."
    wget -O "${archive}" "${url}"
fi

# md5sum is GNU coreutils and is not present on macOS, which ships md5 instead.
# Verify with whichever is available and carry on if neither is.
if command -v md5sum > /dev/null; then
    echo "${md5}  ${archive}" | md5sum -c -
elif command -v md5 > /dev/null; then
    if [ "$(md5 -q "${archive}")" != "${md5}" ]; then
        echo "Error: md5 mismatch for ${archive}" >&2
        exit 1
    fi
    echo "${archive}: OK"
else
    echo "Warning: neither md5sum nor md5 found, skipping checksum verification." >&2
fi

rm -rf "${corpus_dir}"
unzip -q "${archive}" -d "${download_dir}"
rm -f "${archive}"

echo "Successfully downloaded Thorsten-Voice 2022.10."
