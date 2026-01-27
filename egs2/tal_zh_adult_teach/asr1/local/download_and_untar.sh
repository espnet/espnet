#!/usr/bin/env bash

# Manual extractor for TAL Adult Chinese Teaching Speech data.
# The three TAL_ASR-*.zip archives must be downloaded manually from
# https://ai.100tal.com/openData/voice and placed in the archive directory.

set -euo pipefail

if ! command -v unzip >/dev/null; then
  echo "$0: unzip is not installed."
fi

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  cat <<EOF
Usage: local/download_and_untar.sh <extract-dir> [<zip-dir>]

<extract-dir> : Destination directory for the extracted TAL ASR folders.
<zip-dir>     : Location of TAL_ASR-*.zip archives. Defaults to <extract-dir>.
                If provided but missing, the script falls back to <extract-dir>.
EOF
fi

extract_dir=$(realpath $1)
zip_dir=$(realpath ${2:-$extract_dir})

mkdir -p "$extract_dir"

# TAL_ASR-1.zip
if [ -f "$extract_dir/aisolution_data/.complete" ]; then
  echo "$0: aisolution_data already prepared, skipping."
elif [ -f "$zip_dir/TAL_ASR-1.zip" ]; then
  echo "$0: extracting TAL_ASR-1.zip into $extract_dir"
  unzip -q "$zip_dir/TAL_ASR-1.zip" -d "$extract_dir"
  touch "$extract_dir/aisolution_data/.complete"
else
  echo "$0: aisolution_data not found; please download TAL_ASR-1.zip manually from https://ai.100tal.com/openData/voice."
fi

# TAL_ASR-2.zip
if [ -f "$extract_dir/CH/.complete" ]; then
  echo "$0: CH already prepared, skipping."
elif [ -f "$zip_dir/TAL_ASR-2.zip" ]; then
  echo "$0: extracting TAL_ASR-2.zip into $extract_dir"
  unzip -q "$zip_dir/TAL_ASR-2.zip" -d "$extract_dir"
  touch "$extract_dir/CH/.complete"
else
  echo "$0: CH not found; please download TAL_ASR-2.zip manually from https://ai.100tal.com/openData/voice."
fi

# TAL_ASR-3.zip
if [ -f "$extract_dir/MA/.complete" ]; then
  echo "$0: MA already prepared, skipping."
elif [ -f "$zip_dir/TAL_ASR-3.zip" ]; then
  echo "$0: extracting TAL_ASR-3.zip into $extract_dir"
  unzip -q "$zip_dir/TAL_ASR-3.zip" -d "$extract_dir"
  touch "$extract_dir/MA/.complete"
else
  echo "$0: MA not found; please download TAL_ASR-3.zip manually from https://ai.100tal.com/openData/voice."
fi

echo "$0: TAL ASR dataset is ready under $extract_dir"
