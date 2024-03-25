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
cmd=run.pl
# NOTE, use absolute paths !
chime8_root=/raid/users/popcornell/CHiME6/tmp_chimeutils/chime8_dasr_w_gt
download_dir=${PWD}/datasets # where do you want your datasets downloaded
# mixer6 has to be obtained through LDC see official data page
mixer6_root=/raid/users/popcornell/mixer6


# DATAPREP CONFIG
manifests_root=${PWD}/data/lhotse # dir where to save lhotse manifests


# DIARIZATION config
diarization_backend=pyannote
pyannote_access_token=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 # will remove after the challenge.
# inference
diarization_dir=exp/diarization
pyan_merge_closer=0.5
min_dur_on=0.5
pyan_max_length_merged=20
pyan_inf_max_batch=32
pyan_use_pretrained=popcornell/pyannote-segmentation-chime6-mixer6
pyan_ft=0
# fine-tune
pyan_finetune_dir=exp/pyannote_finetuned
pyan_batch_size=32
pyan_learning_rate="1e-5"


# GSS config
ngpu=1
gss_max_batch_dur=200
gss_iterations=5
infer_max_segment_length=200
gss_dsets="chime6_dev,dipco_dev,mixer6_dev,notsofar1_dev"
asr_tt_set="kaldi/chime6/dev/gss kaldi/dipco/dev/gss kaldi/mixer6/dev/gss kaldi/notsofar1/dev/gss"

# ASR config
use_pretrained=
run_on="dev"

if [ -z $use_pretrained ]; then
  echo "Pretrained ASR model not set. Using default !"
  use_pretrained=popcornell/chime7_task1_asr1_baseline
fi

gss_asr_stage=
gss_asr_stop_stage=10

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

if [ -z "$gss_asr_stage" ]; then
  gss_asr_stage=2
fi
# shellcheck disable=SC2207
devices=($(echo $CUDA_VISIBLE_DEVICES | tr "," " "))
if [ "${#devices[@]}" -lt $ngpu ]; then
  echo "Number of available GPUs is less than requested ! exiting. Please check your CUDA_VISIBLE_DEVICES"
  exit
fi





if [[ $run_on != "dev" ]] && [[ $run_on != "eval" ]];
then
  log "run_on argument should be either dev, eval here. ASR training is done in asr1 recipe.
  To fine-tune the pyannote segmentation model you should unset 'pyan_use_pretrained' variable in this script"
  exit
fi



if [ ${stage} -le 0 ] && [ $stop_stage -ge 0 ]; then
  # this script creates the task1 dataset
  gen_splits=train,dev
  if [ $run_on == "eval" ]; then
    gen_splits="eval"
  fi

  if [ -d "$chime8_root" ]; then
    echo "$chime8_root exists already, skipping data download and generation."
  else
    chime-utils dgen dasr $download_dir $mixer6_root $chime8_root --part $gen_splits --download
    # there are also other commands to download the datasets independently see https://github.com/chimechallenge/chime-utils
  fi

fi

if [ ${stage} -le 1 ] && [ $stop_stage -ge 1 ] && [ $diarization_backend == pyannote ] && [ $pyan_ft == 1 ]; then

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

  if [ ! -d ./chime8_dasr ]; then
     ln -s $chime8_root ./chime8_dasr # make link to database.yml also in main dir
  fi

  python local/pyannote_finetune.py --exp_folder $pyan_finetune_dir \
      --batch_size $pyan_batch_size \
      --learning_rate $pyan_learning_rate \
      --token $pyannote_access_token

    prev=pyan_use_pretrained
    pyan_use_pretrained="${PWD}/${pyan_finetune_dir}/lightning_logs/version_0/checkpoints/best.ckpt"
    echo "Using fine-tuned model in $pyan_use_pretrained instead of $prev"
fi

if [ ${stage} -le 2 ] && [ $stop_stage -ge 2 ] && [ $diarization_backend == pyannote ]; then
  # check if pyannote is installed
  if ! python3 -c "import pyannote.audio" &> /dev/null; then
    log "Installing Pyannote Audio."
    (   # not supported 3.0, performance probably will be better with new pipeline
        python3 -m pip install pyannote-audio=="2.1.1"
    )
  fi


  split_gss_dsets_commas=$(echo $gss_dsets | tr "," " ")
  for dset_split in $split_gss_dsets_commas; do
    # for each dataset get the name and part (dev or train)
    dset="$(cut -d'_' -f1 <<<${dset_split})"
    split="$(cut -d'_' -f2 <<<${dset_split})"
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
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # parse all datasets to lhotse
  split_gss_dsets_commas=$(echo $gss_dsets | tr "," " ")
  for dset_split in $split_gss_dsets_commas; do
    # for each dataset get the name and part (dev or train)
    dset="$(cut -d'_' -f1 <<<${dset_split})"
    dset_part="$(cut -d'_' -f2 <<<${dset_split})"
      log "Creating lhotse manifests for ${dset} in ${manifests_root}/${dset}"
      chime-utils lhotse-prep $dset $chime8_root/$dset $manifests_root/${dset}/${dset_part}_orig --txt-norm none --dset-part $dset_part --json-dir ${diarization_dir}/${dset}/${dset_part}
      echo "Discard lhotse supervisions shorter than $min_dur_on, you can change this using --min-dur-on arg"
      chime-utils lhotse-prep discard-length $manifests_root/${dset}/${dset_part}_orig $manifests_root/${dset}/$dset_part --min-len $min_dur_on
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
        --gss-dsets "$gss_dsets" \
        --asr-tt-set "$asr_tt_set" \
        --run-on $run_on --gss-max-batch-dur $gss_max_batch_dur \
        --infer-max-segment-length $infer_max_segment_length \
        --gss-iterations $gss_iterations \
        --run-on-ovrr 1 \
        --cmd "$cmd"

fi
