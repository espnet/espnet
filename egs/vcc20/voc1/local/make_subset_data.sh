#!/usr/bin/env bash

# Make subset files located in data direcoty.

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;


if [ $# -ne 3 ]; then
    echo "Usage: $0 <src_dir> <num_split> <dst_dir>"
    echo "e.g.: $0 data/train_nodev 16 data/train_nodev/split16"
    exit 1
fi

set -eu

src_dir=$1
num_split=$2
dst_dir=$3

[ ! -e "${dst_dir}" ] && mkdir -p "${dst_dir}"

if [ -e "${src_dir}/segments" ]; then
    has_segments=true
    src_segments=${src_dir}/segments
else
    has_segments=false
fi
src_scp=${src_dir}/wav.scp
num_src_utts=$(wc -l < "${src_scp}")

# NOTE: We assume that wav.scp and segments has the same number of lines
if ${has_segments}; then
    num_src_segments=$(wc -l < "${src_segments}")
    if [ "${num_src_segments}" -ne "${num_src_utts}" ]; then
        echo "ERROR: wav.scp and segments has different #lines (${num_src_utts} vs ${num_src_segments})." >&2
        exit 1;
    fi
fi

split_scps=""
for i in $(seq 1 "${num_split}"); do
    split_scps+=" ${dst_dir}/wav.${i}.scp"
done
# shellcheck disable=SC2086
split_scp.pl "${src_scp}" ${split_scps}
if ${has_segments}; then
    split_segments=""
    for i in $(seq 1 "${num_split}"); do
        split_segments+=" ${dst_dir}/segments.${i}"
    done
    # shellcheck disable=SC2086
    split_scp.pl "${src_segments}" ${split_segments}
fi
echo "Successfully make subsets."
