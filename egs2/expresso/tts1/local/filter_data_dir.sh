#!/bin/bash

# This script filters a Kaldi-style data directory to keep only utterances listed in a file.
# Usage: filter_data_dir.sh --utt-list utt_list.txt <srcdir> <destdir>

set -e
set -o pipefail

utt_list=

# Parse options before positional arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --utt-list)
            utt_list="$2"
            shift; shift
            ;;
        *)
            break  # stop parsing options
            ;;
    esac
done

# Check positional arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 --utt-list utt_list.txt <srcdir> <destdir>"
    exit 1
fi

srcdir=$1
destdir=$2

if [ -z "${utt_list}" ]; then
    echo "$0: --utt-list is required"
    exit 1
fi

mkdir -p ${destdir}

# Filter common Kaldi files
for f in wav.scp utt2spk text segments; do
    if [ -f ${srcdir}/$f ]; then
        utils/filter_scp.pl ${utt_list} ${srcdir}/$f > ${destdir}/$f
    fi
done

# Create spk2utt from utt2spk
if [ -f ${destdir}/utt2spk ]; then
    utils/utt2spk_to_spk2utt.pl ${destdir}/utt2spk > ${destdir}/spk2utt
fi

# Copy optional files
for f in spk2gender cmvn.scp reco2file_and_channel; do
    [ -f ${srcdir}/$f ] && cp ${srcdir}/$f ${destdir}/
done

echo "Filtered data directory saved to ${destdir}"
