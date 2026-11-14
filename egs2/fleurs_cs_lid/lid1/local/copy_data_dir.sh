#!/usr/bin/env bash
# Copy a data directory of this recipe. Utterance ids are not grouped by
# language, so the language-as-speaker validation of the template helper does
# not apply; the directories carry one speaker per utterance instead.
set -e
set -u
set -o pipefail

validate_opts=
. utils/parse_options.sh

if [ $# -ne 2 ]; then
    echo "Usage: $0 [--validate_opts <opts>] <srcdir> <destdir>" >&2
    exit 2
fi
srcdir=$1
destdir=$2

# shellcheck disable=SC2086
utils/copy_data_dir.sh --validate_opts "${validate_opts}" "${srcdir}" "${destdir}"
for f in utt2lang utt2langs lang2utt; do
    cp "${srcdir}/${f}" "${destdir}/${f}"
done
