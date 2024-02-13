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
# CHiME-8 Task 1 Diarization + GSS + ASR baseline system script.
######################################################################################

stage=0
stop_stage=100

# NOTE, use absolute paths !
chime8_root=/raid/users/popcornell/CHiME6/tmp_chimeutils/chime8_dasr
download_dir=${PWD}/datasets # where do you want your datasets downloaded
# mixer6 has to be obtained through LDC see official data page
mixer6_root=/raid/users/popcornell/mixer6


# DATAPREP CONFIG
manifests_root=${PWD}/data/lhotse # dir where to save lhotse manifests
cmd_dprep=run.pl


# DIARIZATION config
diarization_backend=pyannote
pyannote_access_token=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 # will remove after the challenge.
# inference
diarization_dir=exp/diarization
diar_inf_dset="dev"
pyan_merge_closer=0.5
pyan_max_length_merged=20
pyan_inf_max_batch=32
pyan_use_pretrained= #popcornell/pyannote-segmentation-chime6-mixer6
download_baseline_diarization=0
# fine-tune
pyan_finetune_dir=exp/pyannote_finetuned
pyan_batch_size=64
pyan_learning_rate="1e-5"


# GSS config
ngpu=2
gss_max_batch_dur=90
gss_iterations=5
infer_max_segment_length=20
gss_dsets="dipco_dev"
asr_tt_set="kaldi/dipco/dev/gss"

# ASR config
use_pretrained=
decode_train="dev"


gss_asr_stage=
gss_asr_stop_stage=10

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

if [ -z "$gss_asr_stage" ]; then
  gss_asr_stage=2
fi


if [ "${decode_train}" == "eval" ]; then
  diar_inf_dset="eval"
fi


if [[ $decode_train != "dev" ]] && [[ $decode_train != "eval" ]];
then
  log "decode_train argument should be either dev, eval here. ASR training is done in asr1 recipe.
  To fine-tune the pyannote segmentation model you should unset 'pyan_use_pretrained' variable in this script"
  exit
fi



if [ $download_baseline_diarization == 1 ]; then
  log "Using organizer-provided JSON manifests from the baseline diarization system."
  if [ ! -d CHiME7DASRDiarizationBaselineJSONs ]; then
      exit #FIXME later provide
      git clone https://github.com/popcornell/CHiME7DASRDiarizationBaselineJSONs
  fi
  mkdir -p exp/diarization
  cp -r CHiME7DASRDiarizationBaselineJSONs/diarization exp/
  stage=3
fi


if [ ${stage} -le 0 ] && [ $stop_stage -ge 0 ]; then
  # this script creates the task1 dataset
  gen_splits=train,dev
  if [ $decode_train == "eval" ]; then
    gen_splits="eval"
  fi

  if [ -d "$chime8_root" ]; then
    echo "$chime8_root exists already, skipping data download and generation."
  else
    chime-utils dgen dasr $download_dir $mixer6_root $chime8_root --part $gen_splits --download
    # there are also other commands to download the datasets independently see https://github.com/chimechallenge/chime-utils
  fi

fi


if [ ${stage} -le 1 ] && [ $stop_stage -ge 1 ] && [ $diarization_backend == pyannote ] && [ -z "${pyan_use_pretrained}" ]; then

  if ! python3 -c "import pyannote.audio" &> /dev/null; then
    log "Installing Pyannote Audio."
    (
        python3 -m pip install pyannote-audio
    )
  fi
  # prep data for fine-tuning the pyannote segmentation model
  # we use only CHiME-6 training set because DiPCo has only development and
  # Mixer 6 training annotation is not suitable right away for diarization
  # training. You can try to leverage it in a semi-supervised way however.
  python local/pyannote_dprep.py -r $chime8_root --output_root data/pyannote_diarization
  # fine-tuning the model
  if [ ! -f database.yml ]; then
     ln -s local/database.yml database.yml # make link to database.yml also in main dir
  fi
  python local/pyannote_finetune.py --exp_folder $pyan_finetune_dir \
      --batch_size $pyan_batch_size \
      --learning_rate $pyan_learning_rate \
      --token $pyannote_access_token

  if [ -z "${pyan_use_pretrained}" ]; then
    # use the one fine-tuned now
    pyan_use_pretrained="${PWD}/${pyan_finetune_dir}/lightning_logs/version_0/checkpoints/best.ckpt"
  fi
fi

if [ ${stage} -le 2 ] && [ $stop_stage -ge 2 ] && [ $diarization_backend == pyannote ]; then
  # check if pyannote is installed
  if ! python3 -c "import pyannote.audio" &> /dev/null; then
    log "Installing Pyannote Audio."
    (
        python3 -m pip install pyannote-audio=="2.1.1"
    )
  fi

  for dset in dipco; do #FIXME
    for split in $diar_inf_dset; do
        if [ $dset == mixer6 ]; then
          mic_regex="(?!CH01|CH02|CH03)(CH[0-9]+)" # exclude close-talk CH01, CH02, CH03
          sess_regex="([0-9]+_[0-9]+_(LDC|HRM)_[0-9]+)"
        else
          mic_regex="(U[0-9]+)" # exclude close-talk
          sess_regex="(S[0-9]+)"
        fi
        # diarizing with pyannote + ensembling across mics with dover-lap
        python local/pyannote_diarize.py -i ${chime8_root}/${dset}/audio/${split} \
              -o ${diarization_dir}/${dset}/${split} \
              -u ${chime8_root}/${dset}/uem/${split}/all.uem \
              --mic_regex $mic_regex \
              --sess_regex $sess_regex \
              --token ${pyannote_access_token} \
              --segmentation_model $pyan_use_pretrained \
              --merge_closer $pyan_merge_closer \
              --max_length_merged $pyan_max_length_merged \
              --max_batch_size $pyan_inf_max_batch
    done
  done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # parse all datasets to lhotse
  for dset in dipco; do
    for dset_part in $diar_inf_dset; do
      log "Creating lhotse manifests for ${dset} in ${manifests_root}/${dset}"
      chime-utils lhotse-prep $dset $chime8_root/$dset $manifests_root/${dset}/${dset_part}_orig --txt-norm none --dset-part $dset_part --json-dir ${diarization_dir}/${dset}/${dset_part}
      echo "Discard lhotse supervisions shorter than 0.2"
      chime-utils lhotse-prep discard-length $manifests_root/${dset}/${dset_part}_orig $manifests_root/${dset}/$dset_part --min-len 0.2
    done
  done
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  log "Performing GSS+Channel Selection+ASR inference on diarized output"
  # now that we have diarized the dataset, we can run the sub-track 1 baseline
  # and use the diarization output in place of oracle diarization.
  # NOTE that it is supposed you either trained the ASR model or
  # use the pretrained one: popcornell/chime7_task1_asr1_baseline
  ./run_gss_asr.sh --chime8-root $chime8_root --stage $gss_asr_stage \
        --stop-stage $gss_asr_stop_stage --ngpu $ngpu \
        --use-pretrained $use_pretrained \
        --decode-train $decode_train --gss-max-batch-dur $gss_max_batch_dur \
        --infer-max-segment-length $infer_max_segment_length \
        --gss-iterations $gss_iterations \
        --diar-score 1 \
        --gss-dsets $gss_dsets \
        --asr-tt-set $asr_tt_set
fi
