#!/usr/bin/env bash
# Use test sets of another recipe's dump as test sets of this one, e.g. the 21 sets of
# frontia_vad_v1 that its published results are on:
#
#   local/import_test_sets.sh [--suffix _v1] <other recipe>/dump/raw <set> [<set> ...]
#
# Copies the text files of each set (wav.scp, utt2num_samples, rttm, espnet_rttm,
# feats_type, audio_format) into dump/raw/<set><suffix>; the audio stays where the other
# dump's wav.scp points (its entries are absolute paths). An existing set is left as is.
set -euo pipefail
suffix=_v1
. utils/parse_options.sh
src=$1
shift
for s in "$@"; do
    out=dump/raw/${s}${suffix}
    [ -f "${out}/wav.scp" ] && continue
    for f in wav.scp utt2num_samples rttm; do
        [ -f "${src}/${s}/${f}" ] || { echo "$0: no ${src}/${s}/${f}"; exit 1; }
    done
    if grep -qv '^[^ ]* /' "${src}/${s}/wav.scp"; then
        echo "$0: ${src}/${s}/wav.scp has relative paths"; exit 1
    fi
    mkdir -p "${out}"
    for f in wav.scp utt2num_samples rttm espnet_rttm feats_type audio_format; do
        [ -f "${src}/${s}/${f}" ] && cp "${src}/${s}/${f}" "${out}/${f}"
    done
    echo "${out}: $(wc -l < "${out}/wav.scp") recordings"
done
