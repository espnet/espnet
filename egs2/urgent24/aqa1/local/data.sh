#!/usr/bin/env bash
set -euo pipefail

# Scores and audio are prepared externally; see the template README.
for dset in train dev test; do
    for file in utt2spk metric.scp wav.scp; do
        if [ ! -f "data/${dset}/${file}" ]; then
            echo "Missing data/${dset}/${file}; prepare the scored corpus first." >&2
            exit 1
        fi
        sort -o "data/${dset}/${file}" "data/${dset}/${file}"
    done
    # The shared driver requires these only when reference inputs are enabled.
    for file in text ref_wav.scp; do
        if [ -f "data/${dset}/${file}" ]; then
            sort -o "data/${dset}/${file}" "data/${dset}/${file}"
        fi
    done
    utils/utt2spk_to_spk2utt.pl "data/${dset}/utt2spk" > "data/${dset}/spk2utt"
done
