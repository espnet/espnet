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
# CHiME-7 Task 1 SUB-TASK baseline system script: GSS + ASR using oracle diarization
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


gss_dump_root=./exp/gss
ngpu=4  # set equal to the number of GPUs you have, used for GSS and ASR training
train_min_segment_length=1 # discard sub one second examples, they are a lot in chime6
train_max_segment_length=20  # also reduce if you get OOM, here A100 40GB

# GSS CONFIG
gss_max_batch_dur=90 # set accordingly to your GPU VRAM, A100 40GB you can use 360
# if you still get OOM errors for GSS see README.md
cmd_gss=run.pl # change to suit your needs e.g. slurm !
# note with run.pl your GPUs need to be in exclusive mode otherwise it fails
# to go multi-gpu see https://groups.google.com/g/kaldi-help/c/4lih8UKHBoc
gss_dsets="chime6_train,chime6_dev,dipco_dev,mixer6_dev"
top_k=80
# we do not train with mixer 6 training + GSS here, but you can try.

# ASR CONFIG
# NOTE: if you get OOM reduce the batch size in asr_config YAML file
asr_stage=0 # starts at 13 for inference only
asr_dprep_stage=0
bpe_nlsyms="[inaudible],[laughs],[noise]" # in the baseline these are handled by the dataprep
asr_config=conf/tuning/train_asr_transformer_wavlm_lr1e-4_specaugm_accum1_preenc128_warmup20k.yaml
inference_config="conf/decode_asr_transformer.yaml"
inference_asr_model=valid.acc.ave.pth
asr_tt_set="kaldi/chime6/dev/gss_inf kaldi/dipco/dev/gss/ kaldi/mixer6/dev/gss/ kaldi/chime6/eval/gss_inf kaldi/dipco/eval/gss/ kaldi/mixer6/eval/gss/"
lm_config="conf/train_lm.yaml"
use_lm=false
use_word_lm=false
word_vocab_size=65000
nbpe=500
asr_max_epochs=8
# put popcornell/chime7_task1_asr1_baseline if you want to test with pretrained model
use_pretrained=
decode_only=0

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

# ESPNet does not scale parameters with num of GPUs by default, doing it
# here for you
asr_batch_size=$(calc_int 128*$ngpu) # reduce 128 bsz if you get OOMs errors
asr_max_lr=$(calc_float $ngpu/10000.0)
asr_warmup=$(calc_int 40000.0/$ngpu)

if [ $decode_only == 1 ]; then
  # apply gss only on dev
  gss_dsets="chime6_eval,dipco_eval,mixer6_eval"
fi

if [ ${stage} -le 0 ] && [ $stop_stage -ge 0 ]; then
  # this script creates the task1 dataset
  local/gen_task1_data.sh --chime6-root $chime6_root --stage $dprep_stage  --chime7-root $chime7_root \
    --chime5_root "$chime5_root" \
	  --dipco-root $dipco_root \
	  --mixer6-root $mixer6_root \
	  --stage $dprep_stage \
	  --train_cmd "$cmd_dprep" \
	  --gen-eval $gen_eval
fi


if [ ${stage} -le 1 ] && [ $stop_stage -ge 1 ]; then
  # parse all datasets to lhotse
  for dset in chime6 dipco mixer6; do
    for dset_part in train dev "eval"; do
      if [ $dset == dipco ] && [ $dset_part == train ]; then
          continue # dipco has no train set
      fi

      if [ $decode_only == 1 ] && [ $dset_part == train ]; then
        continue
      fi

      log "Creating lhotse manifests for ${dset} in $manifests_root/${dset}"
      python local/get_lhotse_manifests.py -c $chime7_root \
           -d $dset \
           -p $dset_part \
           -o $manifests_root \
           --ignore_shorter 0.2
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
    max_segment_length=2000 # enhance all in inference, training we can drop longer ones
    channels=all # do not set for the other datasets, use all
    if [ ${dset_name} == dipco ] && [ ${dset_part} == train ]; then
      log "DiPCo has no training set! Exiting !"
      exit
    fi

    if [ ${dset_part} == dev ]; then # use only outer mics
      use_selection=1
    else
      use_selection=0
    fi

    if [ ${dset_part} == train ]; then
      max_segment_length=${train_max_segment_length} # we can discard utterances too long based on asr training
    fi

    log "Running Guided Source Separation for ${dset_name}/${dset_part}, results will be in ${gss_dump_root}/${dset_name}/${dset_part}"
    local/run_gss.sh --manifests-dir $manifests_root --dset-name $dset_name \
          --dset-part $dset_part \
          --exp-dir $gss_dump_root \
          --cmd "$cmd_gss" \
          --nj $ngpu \
          --max-segment-length $max_segment_length \
          --max-batch-duration $gss_max_batch_dur \
          --channels $channels \
          --use-selection $use_selection \
          --top-k $top_k
    log "Guided Source Separation processing for ${dset_name}/${dset_part} was successful !"
  done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then

  asr_train_set=kaldi/train_all_mdm_ihm_rvb_gss
  asr_cv_set=kaldi/chime6/dev/gss # use chime only for validation
  # Decoding on dev set because test is blind for now
  # NOTE that ESPNet will not make copies of the original Kaldi manifests
  # e.g. for training and cv, so if you set $train_max_segment_length these
  # will be discarded also from the test set (if the test set is the same as evaluation)
  # you need to make a copy, here we make a copy inside local/data.sh called gss_inf!
  #asr_tt_set+=" kaldi/chime6/dev/ihm kaldi/dipco/dev/ihm/ kaldi/mixer6/dev/ihm/"
  # uncomment if you do want to decode also on close-talk microphones
  # note however that it could be bad because there won't be any separation.

  pretrained_affix=
  if [ -n "$use_pretrained" ]; then
    asr_train_set=kaldi/dev_ihm_all # dummy one, it is not used
    pretrained_affix+="--skip_data_prep false --skip_train true "
    pretrained_affix+="--download_model ${use_pretrained}"
  fi

  # these are args to ASR data prep, done in local/data.sh
  data_opts="--stage $asr_dprep_stage --chime6-root ${chime6_root} --train-set ${asr_train_set}"
  data_opts+=" --manifests-root $manifests_root --gss_dsets $gss_dsets --gss-dump-root $gss_dump_root"
  data_opts+=" --decode-only $decode_only"
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
    --speed_perturb_factors "0.9 1.0 1.1" \
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

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # final scoring
  log "Scoring ASR predictions for CHiME-7 DASR challenge."
  # note, we re-create the asr exp folder here based on asr.sh
  if [ -n "$use_pretrained" ]; then
    asr_exp="exp/${use_pretrained}"
  else
    asr_tag="$(basename "${asr_config}" .yaml)_raw"
    asr_exp="exp/asr_${asr_tag}"
  fi
  inference_tag="$(basename "${inference_config}" .yaml)"
  inference_tag+="_asr_model_$(echo "${inference_asr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

  for tt_dset in $asr_tt_set; do
    split="$(cut -d'/' -f3 <<<${tt_dset})"
    dset_name="$(cut -d'/' -f2 <<<${tt_dset})"
    if [ ${dset_name} == mixer6 ]; then
      regex="([0-9]+_[0-9]+_(LDC|HRM)_[0-9]+)" # different session naming
    else
      regex="(S[0-9]+)"
    fi
    python local/asr2json.py -i ${asr_exp}/${inference_tag}/${tt_dset}/text -o ${asr_exp}/${inference_tag}/chime7dasr_hyp/$split/$dset_name -r $regex

  done
  LOG_OUT=${asr_exp}/${inference_tag}/scoring/scoring.log
  python local/da_wer_scoring.py -s ${asr_exp}/${inference_tag}/chime7dasr_hyp/$split \
     -r $chime7_root -p $split -o ${asr_exp}/${inference_tag}/scoring -d 0 2>&1 | tee $LOG_OUT
fi
