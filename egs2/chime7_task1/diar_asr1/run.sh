#!/usr/bin/env bash
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
dipco_root=${PWD}/../asr1/datasets/dipco # this will be automatically downloaded
mixer6_root=/raid/users/popcornell/mixer6/ # put yours here

# DATAPREP CONFIG
manifests_root=./data/lhotse # dir where to save lhotse manifests
cmd_dprep=run.pl
dprep_stage=0
gen_eval=0 # please not generate eval before release of mixer 6 eval

# DIARIZATION config
diarization_backend=pyannote
pyannote_access_token=
diarization_dir=exp/diarization
diar_inf_dset="dev"

# GSS config
ngpu=4
gss_max_batch_dur=90

# ASR config
use_pretrained=
decode_only=1

gss_asr_stage=
gss_asr_stop_stage=10

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

if [ -z "$gss_asr_stage" ]; then
  gss_asr_stage=2
fi


if [ ${stage} -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Generating CHiME-7 DASR Challenge data."
  # this script creates the task1 dataset
  local/gen_task1_data.sh --chime6-root $chime6_root --stage $dprep_stage  --chime7-root $chime7_root \
    --chime5_root "$chime5_root" \
	  --dipco-root $dipco_root \
	  --mixer6-root $mixer6_root \
	  --stage $dprep_stage \
	  --train_cmd "$cmd_dprep" \
	  --gen-eval $gen_eval
fi


if [ ${stage} -le 1 ] && [ $stop_stage -ge 1 ] && [ $diarization_backend == pyannote ]; then
  # check if pyannote is installed
  if ! python3 -c "import pyannote.audio" &> /dev/null; then
    log "Installing Pyannote Audio."
    (
        python3 -m pip install pyannote-audio
    )
  fi
  # check if dover-lap is installed too
  if ! command -v dover-lap &>/dev/null; then
  log "Installing DOVER-Lap."
  (
        python3 -m pip install dover-lap
        # need intervaltree with merge_neightbours
        python3 -m pip --upgrade --force-reinstall git+https://github.com/chaimleib/intervaltree
  )
  fi

  for dset in chime6 dipco mixer6; do
    for split in $diar_inf_dset; do
        if [ $dset == mixer6 ]; then
          mic_regex="(?!CH01|CH02|CH03)(CH[0-9]+)" # exclude close-talk CH01, CH02, CH03
          sess_regex="([0-9]+_[0-9]+_(LDC|HRM)_[0-9]+)"
        else
          mic_regex="(U[0-9]+)" # exclude close-talk
          sess_regex="(S[0-9]+)"
        fi
        # diarizing with pyannote + ensembling across mics with dover-lap
        python local/pyannote_diarize.py -i ${chime7_root}/${dset}/audio/${split} \
              -o ${diarization_dir}/${dset}/${split} \
              -u ${chime7_root}/${dset}/uem/${split}/all.uem \
              --mic_regex $mic_regex \
              --sess_regex $sess_regex \
              --token ${pyannote_access_token}
    done
  done
fi


if [ ${stage} -le 2 ] && [ $stop_stage -ge 2 ]; then
  # parse all datasets to lhotse
  for dset in chime6 dipco mixer6; do
    for dset_part in $diar_inf_dset; do
      log "Creating lhotse manifests for ${dset} in $manifests_root/${dset}"
      python local/get_lhotse_manifests.py -c $chime7_root \
           -d $dset \
           -p $dset_part \
           -o $manifests_root --diar_jsons_root ${diarization_dir}/${dset} \
           --ignore_shorter 0.5
    done
  done
fi


if [ ${stage} -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Performing GSS+Channel Selection+ASR inference on diarized output"
  # now that we have diarized the dataset, we can run the sub-track 1 baseline
  # and use the diarization output in place of oracle diarization.
  # NOTE that it is supposed you either trained the ASR model or
  # use the pretrained one: popcornell/chime7_task1_asr1_baseline
  ./run_gss_asr.sh --chime7-root $chime7_root --stage $gss_asr_stage \
        --stop-stage $gss_asr_stop_stage --ngpu $ngpu \
        --use-pretrained $use_pretrained \
        --decode_only $decode_only --gss-max-batch-dur $gss_max_batch_dur \
        --diar-score 1
fi
