#!/usr/bin/env bash

# Apache 2.0

# Download and unpack one language of a Common Voice release.
#
# Mozilla does not serve the Common Voice archives from public URLs anymore
# (the old voice-prod-bundler-*.s3.amazonaws.com links return 403): the download
# links are signed and only handed out after the license has been accepted on
# https://commonvoice.mozilla.org/en/datasets. This script therefore
#   1. does nothing if the corpus is already unpacked in <data-dir>,
#   2. downloads <url> if one is given (i.e. the signed link from the website),
#   3. unpacks an archive that was placed in <data-dir> by hand,
#   4. and otherwise explains how to obtain the data.
#
# The archives expand into <corpus>/<lang>/, while the data preparation scripts
# expect the tsv files and clips/ directly in <data-dir>, so the layout of the
# unpacked corpus is normalized here.

set -e
set -u
set -o pipefail

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
  echo "Usage: $0 <data-dir> <lang> <corpus> [<url>]"
  echo "e.g.: $0 downloads/es es cv-corpus-4-2019-12-10"
  echo "      $0 downloads/es es cv-corpus-4-2019-12-10 'https://<signed link>'"
  exit 1
fi

data=$1
lang=$2
corpus=$3
url=${4:-}

archive="${corpus}-${lang}.tar.gz"

if [ -f "${data}/validated.tsv" ] && [ -d "${data}/clips" ]; then
  echo "$0: ${data} already contains ${corpus} (${lang}), skipping the download."
  exit 0
fi

mkdir -p "${data}"

# an archive that has been downloaded by hand, under its current or its legacy name
local_archive=
for f in "${data}/${archive}" "${data}/../${archive}" \
         "${data}/${lang}.tar.gz" "${data}/../${lang}.tar.gz"; do
  if [ -f "${f}" ]; then
    local_archive=${f}
    break
  fi
done

if [ -n "${url}" ]; then
  if ! which wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1
  fi
  echo "$0: downloading ${archive} from ${url}. This may take some time, please be patient."
  # NOTE: the signed links carry a query string, so the output file has to be
  # named explicitly (-O) instead of being derived from the URL.
  if ! wget --no-check-certificate -O "${data}/${archive}.tmp" "${url}"; then
    rm -f "${data}/${archive}.tmp"
    echo "$0: error executing wget ${url}"
    exit 1
  fi
  mv "${data}/${archive}.tmp" "${data}/${archive}"
  local_archive=${data}/${archive}
elif [ -n "${local_archive}" ]; then
  echo "$0: using the already downloaded ${local_archive}"
else
  # cv-corpus-4-2019-12-10 -> 4.0, cv-corpus-5.1-2020-06-22 -> 5.1
  hf_version=$(echo "${corpus}" | sed -E 's/^cv-corpus-([0-9]+(\.[0-9]+)?).*/\1/')
  case "${hf_version}" in
    *.*) ;;
    *) hf_version="${hf_version}.0" ;;
  esac
  cat <<MESSAGE
$0: ${corpus} (${lang}) cannot be downloaded automatically.
Mozilla does not distribute Common Voice through public URLs anymore, so the
corpus has to be obtained manually (its terms have to be accepted once):
  1. open https://commonvoice.mozilla.org/en/datasets
  2. select the release "${corpus}" and the language of "${lang}",
     then accept the terms to get a download link
  3. either pass that link to the recipe (quote it, it contains '&'):
       ./local/data.sh --cv_data_url '<link>'
     or download ${archive} yourself and put it in ${data}/
The same data is also available, after accepting the terms, from
  https://huggingface.co/datasets/mozilla-foundation/common_voice_${hf_version/./_}
MESSAGE
  exit 1
fi

echo "$0: unpacking ${local_archive}"
tar -xzf "${local_archive}" -C "${data}"

# ${data}/${corpus}/${lang}/* (or ${data}/${corpus}/*) -> ${data}/*
for d in "${data}/${corpus}/${lang}" "${data}/${corpus}"; do
  if [ -f "${d}/validated.tsv" ]; then
    mv "${d}"/* "${data}/"
    rmdir "${d}" 2>/dev/null || true
    rmdir "${data}/${corpus}" 2>/dev/null || true
    break
  fi
done

if [ ! -f "${data}/validated.tsv" ]; then
  echo "$0: ${data}/validated.tsv is missing after unpacking ${local_archive}"
  exit 1
fi

echo "$0: successfully prepared ${corpus} (${lang}) in ${data}"
