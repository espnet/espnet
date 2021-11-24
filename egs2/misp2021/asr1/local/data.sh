#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=0
stop_stage=1
log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;



if [ ! -e "${MISP2021}" ]; then
    log "Fill the value of 'MISP2021' of db.sh"
    exit 1
fi


enhancement_dir=data/misp2021_far_WPE

###########################################################################
# wpe+beamformit
###########################################################################
# use nara-wpe and beamformit to enhance multichannel misp data
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  log "stage 0: Nara-wpe and Beamformit"
  for x in dev train ; do
    local/enhancement.sh $MISP2021/audio/$x ${enhancement_dir}/audio/$x  || exit 1;
  done
fi

###########################################################################
# prepare data
###########################################################################

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  for x in dev train ; do
    if [[ ! -f data/${x}_far/.done ]]; then
      local/prepare_data.sh $MISP2021 $enhancement_dir $x data/${x}_far || exit 1;
    fi
  done
fi
