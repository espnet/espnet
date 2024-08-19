#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
help_message=$(cat << EOF
Usage: $0

Options:
    --remove_archive (bool): true or false
      With remove_archive=True, the archives will be removed after being successfully downloaded and un-tarred.
EOF
)
SECONDS=0

# Data preparation related
data_url=http://lingtools.uoregon.edu/coraal/coraal_download_list.txt
remove_archive=false
download_opt=


log "$0 $*"


. ./utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -gt 1 ]; then
  log "${help_message}"
  exit 2
fi

if "$remove_archive"; then
  download_opt="--remove-archive"
fi

if [ -z "${CORAAL}" ]; then
  log "Error: \$CORAAL is not set in db.sh."
  exit 2
fi


log "Download data to ${CORAAL}"
if [ ! -d "${CORAAL}" ]; then
    mkdir -p "${CORAAL}"
fi
# To absolute path
CORAAL=$(cd ${CORAAL}; pwd)

echo local/download_and_untar.sh ${download_opt} "${CORAAL}" "${data_url}"
local/download_and_untar.sh ${download_opt} "${CORAAL}" "${data_url}"

coraal_audio_dir=${CORAAL}
coraal_text=${CORAAL}/transcript.tsv


log "Data Preparation"

log "Generate segments and transcript"
echo python3 local/snippet_generation.py "${CORAAL}" "${CORAAL}" 0.1 30
python3 local/snippet_generation.py "${CORAAL}" "${CORAAL}" 0.1 30
