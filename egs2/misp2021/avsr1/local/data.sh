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
stop_stage=3
log "$0 $*"
. utils/parse_options.sh

nj=6

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

if [[ ! -f local/lrw_resnet18_mstcn.pth.tar ]]; then
    log "You need to download lrw_resnet18_mstcn.pth.tar from https://bit.ly/3glF4k5 or https://bit.ly/3513Ror (key: um1q) and put the pretrained model to local/"
    exit 1
fi

if [ ! -d extractor ]; then
    git clone https://github.com/mispchallenge/misp2021_baseline.git
    mv misp2021_baseline/task2_avsr_nn_hmm/extractor/ extractor/
    rm -rf misp2021_baseline
    ln -s $PWD/local/lrw_resnet18_mstcn.pth.tar extractor/models/lrw_resnet18_mstcn.pth.tar
fi


enhancement_dir=data/misp2021_far_WPE
data_roi=$MISP2021/roi

###########################################################################
# wpe+beamformit
###########################################################################
# use nara-wpe and beamformit to enhance multichannel misp data
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  log "stage 0: Nara-wpe and Beamformit"
  for x in dev train ; do
    local/enhancement.sh --nj ${nj} $MISP2021/audio/$x ${enhancement_dir}/audio/$x  || exit 1;
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


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  for x in dev_far train_far; do
    if [ ! -f data/$x/mfcc.done ]; then
      mfccdir=mfcc
      steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj $nj data/$x exp/make_mfcc/$x $mfccdir
      utils/fix_data_dir.sh data/$x
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
      utils/fix_data_dir.sh data/$x
      touch data/$x/mfcc.done
    fi
  done
fi

###########################################################################
# prepare video data
###########################################################################

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # extract visual ROI, store as npz (item: data); extract visual embedding; concatenate visual embedding and mfcc
  for x in dev_far train_far ; do
    local/extract_far_video_roi.sh  --nj ${nj} data/${x} $data_roi/${x} data/${x} || exit 1;
  done
fi
