# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail
calc_float() { awk "BEGIN{ printf \"%.12f\n\", $* }"; }
calc_int() { awk "BEGIN{ printf \"%.d\n\", $* }"; }
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
######################################################################################
# CHiME-7 Task 1 MAIN TRACK baseline system script: GSS + ASR using EEND diarization
######################################################################################

stage=0
stop_stage=100

# NOTE, use absolute paths !
chime7_root=${PWD}/chime7_task1
chime5_root= # you can leave it empty if you have already generated CHiME-6 data
chime6_root=/raid/users/popcornell/CHiME6/espnet/egs2/chime6/asr1/CHiME6 # will be created automatically from chime5
# but if you have it already it will be skipped, please put your own path
dipco_root=${PWD}/datasets/dipco # this will be automatically downloaded
mixer6_root=/raid/users/popcornell/mixer6/ # put yours here

# DATAPREP CONFIG
manifests_root=./data/lhotse # dir where to save lhotse manifests
cmd_dprep=run.pl
dprep_stage=0
gen_eval=0 # please not generate eval before release of mixer 6 eval

asr_use_pretrained=
asr_decode_only=0

. ./path.sh#!/usr/bin/env bash

. ./cmd.sh
. ./utils/parse_options.sh


if [ ${stage} -le 0 ] && [ $stop_stage -ge 0 ]; then
  log("Generating CHiME-7 DASR Challenge data.")
  # this script creates the task1 dataset
  local/gen_task1_data.sh --chime6-root $chime6_root --stage $dprep_stage  --chime7-root $chime7_root \
    --chime5_root "$chime5_root" \
	  --dipco-root $dipco_root \
	  --mixer6-root $mixer6_root \
	  --stage $dprep_stage \
	  --train_cmd "$cmd_dprep" \
	  --gen-eval $gen_eval
fi


# git clone diarization system here if it does not exist
if ! [ -d "./vader" ]; then
   log("Getting the diarization baseline codebase")
   git clone -b chime7dasr https://github.com/popcornell/vader
fi


if [ ${stage} -le 1 ] && [ $stop_stage -ge 1 ]; then
  # optional training
  python vader/train.py --train_manifests

fi


if [ ${stage} -le 3 ] && [ $stop_stage -ge 3 ]; then
  log("Performing GSS+Channel Selection+ASR inference on diarized output")
  # now that we have diarized the dataset, we can run the sub-track 1 baseline
  # and use the diarization output in place of oracle diarization.
  ./../asr1/run.sh --chime7-root $chime7_root --stage 2 --ngpu $ngpu \
        --use-pretrained $asr_use_pretrained \
        --decode_only $asr_decode_only --gss-max-batch-dur $gss_max_batch_dur
fi