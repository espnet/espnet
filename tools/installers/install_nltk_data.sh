#!/usr/bin/env bash
set -euo pipefail

#   install_nltk_data.sh <package> <download dir>
#
# Not `python -m nltk.downloader`. That CLI calls download() with
# halt_on_error=False, and on a failed download it prompts and reads stdin:
#
#   [nltk_data] Error loading averaged_perceptron_tagger_eng: <urlopen
#   [nltk_data]     error ...: no validated address for host
#   [nltk_data]     'raw.githubusercontent.com'
#   Error installing package. Retry? [n/y/e]
#   EOFError: EOF when reading a line
#
# So in CI a transient network failure arrives as an EOFError traceback from
# input() with the real cause scrolled off above it, and it stops the build on
# the first attempt - there is no retry in nltk at all. That took the macOS
# job down on master. raise_on_error=True raises before the prompt is reached;
# the loop below is what makes it a retry.

package="${1:?usage: $0 <nltk package> <download dir>}"
directory="${2:?usage: $0 <nltk package> <download dir>}"

attempts=3
wait=5
for attempt in $(seq 1 "${attempts}"); do
    if python -c '
import sys

import nltk

nltk.download(sys.argv[1], download_dir=sys.argv[2], raise_on_error=True)
' "${package}" "${directory}"; then
        exit 0
    fi
    echo "Attempt ${attempt}/${attempts} failed for nltk package ${package}"
    if [ "${attempt}" -lt "${attempts}" ]; then
        echo "Waiting ${wait}s before retry..."
        sleep "${wait}"
        wait=$((wait * 2))
    fi
done

echo "Could not download the nltk package ${package} into ${directory}" >&2
exit 1
