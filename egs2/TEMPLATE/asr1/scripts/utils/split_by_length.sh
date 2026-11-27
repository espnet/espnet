#!/usr/bin/env bash

# Copyright 2026 Carnegie Mellon University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Split a key file into parts of about equal total length.
#
# With --split_by_length true, and when every key of <key_file> has a length in
# <lengths_file> (normally utt2num_samples), the parts are made by
# pyscripts/utils/split_scp_by_length.py: longest entries first, each to the part
# with the smallest total length, every part in the order of <key_file>.
# Otherwise (--split_by_length false, no lengths file, or a lengths file that
# misses keys, e.g. with non-raw feats_type or text-only data) the parts are made
# by utils/split_scp.pl, i.e. contiguous blocks of equal line counts.
#
# Only the assignment of keys to parts differs between the two; callers must
# merge the outputs of the parts independently of their order (sort by key).

set -euo pipefail

split_by_length=true  # Split by total length when <lengths_file> covers all keys.
python=python3        # Python to run split_scp_by_length.py.

help_message=$(cat << EOF
Usage: $0 [options] <key_file> <lengths_file> <out1> [<out2> ...]
e.g.: $0 data/train/wav.scp data/train/utt2num_samples log/train.1.scp log/train.2.scp

Options:
    --split_by_length  # Split by total length instead of line count (default="${split_by_length}").
    --python           # Python to run split_scp_by_length.py (default="${python}").
EOF
)

. ./utils/parse_options.sh

if [ $# -lt 3 ]; then
    echo "${help_message}" >&2
    exit 1
fi

key_file=$1
lengths_file=$2
shift 2

if [ "${split_by_length}" != true ] && [ "${split_by_length}" != false ]; then
    echo "$0: --split_by_length must be true or false, got '${split_by_length}'" >&2
    exit 1
fi

if "${split_by_length}"; then
    if [ ! -s "${lengths_file}" ]; then
        echo "$0: ${lengths_file} not found; splitting ${key_file} by line count" >&2
        split_by_length=false
    else
        # Number of keys of key_file that have no length in lengths_file.
        num_missing=$(LC_ALL=C awk '
            NR == FNR { if (NF >= 2) len[$1] = 1; next }
            NF > 0 && !($1 in len) { n++ }
            END { print n + 0 }
        ' "${lengths_file}" "${key_file}")
        if [ "${num_missing}" -gt 0 ]; then
            echo "$0: ${num_missing} keys of ${key_file} have no length in ${lengths_file};" \
                "splitting by line count" >&2
            split_by_length=false
        fi
    fi
fi

if "${split_by_length}"; then
    "${python}" pyscripts/utils/split_scp_by_length.py \
        --lengths "${lengths_file}" "${key_file}" "$@"
else
    utils/split_scp.pl "${key_file}" "$@"
fi
