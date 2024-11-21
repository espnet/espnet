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
# CHiME-8 Task 1 GSS + ASR baseline system script.
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

gss_dump_root=./exp/gss
ngpu=3  # set equal to the number of GPUs you have, used for GSS and ASR training
train_min_segment_length=1 # discard sub one second examples, they are a lot in chime6
train_max_segment_length=20  # also reduce if you get OOM, here we used A100 40GB
infer_max_segment_length=200

# GSS CONFIG
use_selection=1 # always use selection
gss_max_batch_dur=200 # set accordingly to your GPU VRAM, A100 40GB you can use 360
# if you still get OOM errors for GSS see README.md
cmd=run.pl # change to suit your needs e.g. slurm !
# NOTE !!! with run.pl your GPUs need to be in exclusive mode otherwise it fails
# to go multi-gpu see https://groups.google.com/g/kaldi-help/c/4lih8UKHBoc
gss_dsets="chime6_train,mixer6_train,dipco_train,notsofar1_train,chime6_dev,dipco_dev,mixer6_dev"
gss_iterations=5
top_k=80
# we do not train with mixer 6 training + GSS here, but you can try.

# ASR CONFIG
# NOTE: if you get OOM reduce the batch size in asr_config YAML file
asr_tag=chime8_ebranchformer # name of the ASR model you want to train
asr_stage=0 # starts at 13 for inference only
asr_dprep_stage=0
bpe_nlsyms="[inaudible],[laughs],[noise]" # in the baseline these are handled by the dataprep
asr_config="conf/tuning/train_asr_ebranchformer_wavlm_lr1e-4_specaugm_accum1_preenc128_warmup40k.yaml"
inference_config="conf/decode_asr_transformer.yaml"
inference_asr_model=valid.acc.ave.pth
asr_train_set=kaldi/train_all_mdm_ihm_rvb_gss
asr_cv_set=kaldi/chime6/dev/gss # we used only chime6 but maybe you can use a combination of all

# note we use notsofar training here because dev gt is not available
asr_tt_set="kaldi/chime6/dev/gss kaldi/dipco/dev/gss/ kaldi/mixer6/dev/gss/"
lm_config="conf/train_lm.yaml"
use_lm=false
use_word_lm=false
word_vocab_size=65000
nbpe=500
asr_max_epochs=8
# put popcornell/chime7_task1_asr1_baseline if you want to test with pretrained model
use_pretrained=
run_on="dev" # chose from dev, train, test
run_on_ovrr=0 # set to one if you want to bypass run_on overriding gss_dsets and asr_tt_set

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh


if [[ $run_on != "dev" ]] && [[ $run_on != "eval" ]] && [[ "$run_on" != "train" ]]; then
  log "run_on argument should be either dev, eval, train or val"
  exit
fi

if [ "$run_on" == "train" ] && [ -n "$use_pretrained" ]; then
  log "You cannot pass a pretrained model and also ask this script to do training from scratch with --run-on train."
  log "You are asking to use $use_pretrained pretrained model."
  exit
fi


if [ $run_on == "dev" ] && [ $run_on_ovrr == 0 ]; then
  # apply gss only on dev
  gss_dsets="chime6_dev,dipco_dev,mixer6_dev"
  asr_tt_set="kaldi/chime6/dev/gss kaldi/dipco/dev/gss kaldi/mixer6/dev/gss"
elif
  [ $run_on == "eval" ] && [ $run_on_ovrr == 0 ]; then
  # apply gss only on eval
  gss_dsets="chime6_eval,dipco_eval,mixer6_eval,notsofar1_eval"
  asr_tt_set="kaldi/chime6/eval/gss kaldi/dipco/eval/gss/ kaldi/mixer6/eval/gss/ kaldi/notsofar1/eval/gss"
fi


# shellcheck disable=SC2207
devices=($(echo $CUDA_VISIBLE_DEVICES | tr "," " "))
if [ "${#devices[@]}" -lt $ngpu ]; then
  echo "Number of available GPUs is less than requested ! exiting. Please check your CUDA_VISIBLE_DEVICES"
  exit
fi


# ESPNet does not scale parameters with num of GPUs by default, doing it
# here for you
asr_batch_size=$(calc_int 128*$ngpu) # reduce 128 bsz if you get OOMs errors
asr_max_lr=$(calc_float $ngpu/10000.0)
asr_warmup=$(calc_int 40000.0/$ngpu)



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


if [ ${stage} -le 1 ] && [ $stop_stage -ge 1 ]; then
  # parse all datasets to lhotse
  for dset in chime6 dipco notsofar1 mixer6; do
    if [ "$run_on" == "eval" ]; then
      dset_part="eval"
    elif [ "$run_on" != "eval" ]; then
      if [ "$dset" == "mixer6" ]; then
        dset_part="train_call train_intv train dev"
      else
        dset_part="train dev"
      fi
    fi

     for c_part in $dset_part; do
        if [[ $c_part = @(train|train_call|train_intv) ]]; then
          txt_norm="chime8" # in training use the chime8, whisper-style normalization
          mics="ihm,mdm"
        elif [[ $run_on == "train" ]]; then
          txt_norm="chime8" # in train apply normalization
          mics="mdm" # ihm not available for dev
        else
          txt_norm="none" # text norm hurts GSS with oracle diarization, in inference we do not use.
          mics="mdm" # ihm not available for dev
        fi

        log "Creating lhotse manifests for ${dset}, $c_part set $mics microphones, in $manifests_root/${dset}"
        # no text norm here because it hurts GSS
        chime-utils lhotse-prep $dset $chime8_root/$dset $manifests_root/${dset}/${c_part}_orig --dset-part $c_part --txt-norm "$txt_norm" -m "$mics"
        echo "Discard lhotse supervisions shorter than 0.2" # otherwise probelms with WavLM
        chime-utils lhotse-prep discard-length $manifests_root/${dset}/${c_part}_orig $manifests_root/${dset}/$c_part --min-len 0.2
     done
  done

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # check if GSS is installed, if not stop, user must manually install it
  ! command -v gss &>/dev/null && log "GPU-based Guided Source Separation (GSS) could not be found,
  #    please refer to the README for how to install it. \n
  #    See also https://github.com/desh2608/gss for more informations." && exit 1;

  split_gss_dsets_commas=$(echo $gss_dsets | tr "," " ")
  for dset in $split_gss_dsets_commas; do
    # for each dataset get the name and part (dev or train)
    dset_name="$(cut -d'_' -f1 <<<${dset})"
    dset_part="$(cut -d'_' -f2 <<<${dset})"
    max_segment_length=${infer_max_segment_length}
    channels=all

    subpart="$(cut -d'_' -f3 <<<${dset})"
    if [ $dset_name == "mixer6" ] && [ $dset_part == "train" ] && [ -n "$subpart" ]; then
      dset_part="${dset_part}_${subpart}" # allow for train_call, train_intv for mixer6
    fi

    log "Running Guided Source Separation for ${dset_name}/${dset_part}, results will be in ${gss_dump_root}/${dset_name}/${dset_part}"
    # shellcheck disable=SC2039
    local/run_gss.sh --manifests-dir $manifests_root --dset-name $dset_name \
          --dset-part $dset_part \
          --exp-dir $gss_dump_root \
          --cmd "$cmd" \
          --nj $ngpu \
          --max-segment-length $max_segment_length \
          --max-batch-duration $gss_max_batch_dur \
          --channels $channels \
          --use-selection $use_selection \
          --top-k $top_k \
          --gss-iterations $gss_iterations
    log "Guided Source Separation processing for ${dset_name}/${dset_part} was successful !"
  done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Decoding on dev set because test is blind for now

  pretrained_affix=
  if [ -n "$use_pretrained" ]; then
    pretrained_affix+="--skip_data_prep false --skip_train true "
    pretrained_affix+="--download_model ${use_pretrained}"
  else
    pretrained_affix+="--asr-tag ${asr_tag}"
  fi


  if [ -z $run_on_ovrr ] || [ $run_on != "train" ]; then
    asr_dprep_stage=3
  fi

  # these are args to ASR data prep, done in local/data.sh
  data_opts="--stage $asr_dprep_stage --dasr-root ${chime8_root} --train-set ${asr_train_set}"
  data_opts+=" --manifests-root $manifests_root"
  data_opts+=" --gss-dump $gss_dump_root --decode-train $run_on --gss-dsets $gss_dsets"
  # override ASR conf/tuning to scale automatically with num of GPUs
  asr_args="--batch_size ${asr_batch_size} --scheduler_conf warmup_steps=${asr_warmup}"
  asr_args+=" --max_epoch=${asr_max_epochs} --optim_conf lr=${asr_max_lr}"

  ./asr.sh \
    --lang en \
    --local_data_opts "${data_opts}" \
    --stage $asr_stage \
    --ngpu $ngpu \
    --asr_args "${asr_args[@]}" \
    --token_type bpe \
    --nbpe $nbpe \
    --bpe_nlsyms "${bpe_nlsyms}" \
    --nlsyms_txt "data/nlsyms.txt" \
    --feats_type raw \
    --feats_normalize utterance_mvn \
    --audio_format "flac" \
    --min_wav_duration $train_min_segment_length \
    --max_wav_duration $train_max_segment_length \
    --speed_perturb_factors "0.95 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --use_lm ${use_lm} \
    --lm_config "${lm_config}" \
    --inference_asr_model ${inference_asr_model} \
    --use_word_lm ${use_word_lm} \
    --word_vocab_size ${word_vocab_size} \
    --train_set "${asr_train_set}" \
    --valid_set "${asr_cv_set}" \
    --test_sets "${asr_tt_set}" \
    --bpe_train_text "data/${asr_train_set}/text" \
    --lm_train_text "data/${asr_train_set}/text" ${pretrained_affix}
fi

if [ "${run_on}" == "eval" ]; then
  log "Scoring not available for eval set till the end of the challenge."
  exit
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # final scoring
  log "Scoring ASR predictions for CHiME-8 DASR challenge."
  # note, we re-create the asr exp folder here based on asr.sh
  if [ -n "$use_pretrained" ]; then
    asr_exp="exp/${use_pretrained}"
  else
    asr_exp="exp/asr_${asr_tag}"
  fi
  inference_tag="$(basename "${inference_config}" .yaml)"
  inference_tag+="_asr_model_$(echo "${inference_asr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

  # when a dataset is both in cv and tt in ESPNet2, the tt dataset is
  # placed unmodified into org. we create a symbolic link
  # so it can be parsed as the other datasets.
  for tt_dset in $asr_tt_set; do
      if [ ! -e "${asr_exp}/${inference_tag}/${tt_dset}" ] && [ -d  "${asr_exp}/${inference_tag}/org/${tt_dset}" ]; then
        # Creating the parent directory
        mkdir -p "${asr_exp}/${inference_tag}/${tt_dset}" && rmdir "${asr_exp}/${inference_tag}/${tt_dset}"
        ln -sf "$(cd ${asr_exp}/${inference_tag}/org/${tt_dset}; pwd)" "${asr_exp}/${inference_tag}/${tt_dset}"
      fi
  done

  for tt_dset in $asr_tt_set; do
    split="$(cut -d'/' -f3 <<<${tt_dset})"
    dset_name="$(cut -d'/' -f2 <<<${tt_dset})"
    if [ ${dset_name} == mixer6 ]; then
      regex="([0-9]+_[0-9]+_(LDC|HRM)_[0-9]+)" # different session naming
    else
      regex="(S[0-9]+)"
    fi
    python local/asr2json.py -i ${asr_exp}/${inference_tag}/${tt_dset}/text -o ${asr_exp}/${inference_tag}/chime8dasr_hyp/$split/$dset_name -r $regex
    # the content of this output folder is what you should send for evaluation to the
    # organizers.
  done

  if [[ $run_on == "train" ]]; then
    split=dev # reset split to dev here
  else
    split=$run_on
  fi

  mkdir -p ${asr_exp}/${inference_tag}/scoring/tcpwer/
  LOG_OUT=${asr_exp}/${inference_tag}/scoring/tcpwer/scoring.log
  chime-utils score tcpwer -s ${asr_exp}/${inference_tag}/chime8dasr_hyp -r $chime8_root \
        --dset-part $split \
        --output-folder ${asr_exp}/${inference_tag}/scoring/tcpwer \
        --ignore-missing 2>&1 | tee $LOG_OUT

  mkdir -p ${asr_exp}/${inference_tag}/scoring/cpwer/
  LOG_OUT=${asr_exp}/${inference_tag}/scoring/cpwer/scoring.log
  chime-utils score cpwer -s  ${asr_exp}/${inference_tag}/chime8dasr_hyp -r $chime8_root \
        --dset-part $split \
        --output-folder ${asr_exp}/${inference_tag}/scoring/cpwer \
        --ignore-missing 2>&1 | tee $LOG_OUT
fi
