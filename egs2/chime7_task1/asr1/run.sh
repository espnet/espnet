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
# CHiME-7 Task 1 SUB-TASK 1 baseline system script: GSS + ASR using oracle diarization
######################################################################################

stage=3
stop_stage=100

# NOTE, use absolute paths !
chime7_root=${PWD}/chime7_task1
chime5_root= # you can leave it empty if you have already generated CHiME-6 data
chime6_root=/raid/users/popcornell/CHiME6/espnet/egs2/chime6/asr1/CHiME6 # will be created automatically from chime5
# but if you have it already it will be skipped
dipco_root=${PWD}/../../chime7/task1/datasets/dipco # this will be automatically downloaded
mixer6_root=/raid/users/popcornell/mixer6/

# DATAPREP CONFIG
manifests_root=./data/lhotse # dir where to save lhotse manifests
cmd_dprep=run.pl
# note with run.pl your GPUs need to be in exclusive mode otherwise it fails
# to go multi-gpu see https://groups.google.com/g/kaldi-help/c/4lih8UKHBoc
dprep_stage=0
gss_dump_root=./exp/gss
ngpu=4  # set equal to the number of GPUs you have, used for GSS and ASR training
train_min_segment_length=1 # discard sub one second examples, they are a lot in chime6
train_max_segment_length=20  # also reduce if you get OOM, here A100 40GB

# GSS CONFIG
gss_max_batch_dur=360 # set accordingly to your GPU VRAM, here A100 40GB
cmd_gss=run.pl # change to suit your needs e.g. slurm !
gss_dsets="chime6_train chime6_dev dipco_dev mixer6_dev" # no mixer6 train in baseline


# ASR CONFIG
# NOTE: if you get OOM reduce the batch size in asr_config YAML file
asr_stage=0 # starts at 13 for inference only
bpe_nlsyms="" # in the baseline these are handled by the dataprep
asr_config=conf/tuning/train_asr_transformer_wavlm_lr1e-4_specaugm_accum1_preenc128_warmup20k.yaml
inference_config="conf/decode_asr_transformer.yaml"
lm_config="conf/train_lm.yaml"
use_lm=false
use_word_lm=false
word_vocab_size=65000
nbpe=500


# and not contribute much (but you may use all)
asr_max_epochs=8
# ESPNet does not scale parameters with num of GPUs by default, doing it
# here for you

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

asr_batch_size=$(calc_int 128*$ngpu) # reduce 128 bsz if you get OOMs errors
asr_max_lr=$(calc_float $ngpu/10000.0)
asr_warmup=$(calc_int 40000.0/$ngpu)


if [ ${stage} -le 0 ] && [ $stop_stage -ge 0 ]; then
  # this script creates the task1 dataset
  local/gen_task1_data.sh --chime6-root $chime6_root --stage $dprep_stage  --chime7-root $chime7_root \
    --chime5_root $chime5_root \
	  --dipco-root $dipco_root \
	  --mixer6-root $mixer6_root \
	  --stage $dprep_stage \
	  --train_cmd $cmd_dprep
fi


if [ ${stage} -le 1 ] && [ $stop_stage -ge 1 ]; then
  # parse all datasets to lhotse
  for dset in chime6 dipco mixer6; do
    for dset_part in train dev; do
      if [ $dset == dipco ] && [ $dset_part == train ]; then
          continue # dipco has no train set
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

  for dset in $gss_dsets; do
    # for each dataset get the name and part (dev or train)
    dset_name="$(cut -d'_' -f1 <<<${dset})"
    dset_part="$(cut -d'_' -f2 <<<${dset})"
    max_segment_length=2000 # enhance all in inference, training we can drop longer ones
    channels=all # do not set for the other datasets, use all
    if [ ${dset_name} == dipco ] && [ ${dset_part} == train ]; then
      log "DiPCo has no training set! Exiting !"
      exit
    fi

    if [ ${dset_name} == dipco ]; then
      channels=2,5,9,12,16,19,23,26,30,33 # in dipco only using opposite mics on each array, works better
    fi

    if [ ${dset_part} == train ]; then
      max_segment_length=${train_max_segment_length} # we can discard utterances too long based on asr training
    fi

    log "Running Guided Source Separation for ${dset_name}/${dset_part}, results will be in ${gss_dump_root}/${dset_name}/${dset_part}"
    local/run_gss.sh --manifests-dir $manifests_root --dset-name $dset_name \
          --dset-part $dset_part \
          --exp-dir $gss_dump_root \
          --cmd $cmd_gss \
          --nj $ngpu \
          --max-segment-length $max_segment_length \
          --max-batch-duration $gss_max_batch_dur \
          --channels $channels
    log "Guided Source Separation processing for ${dset_name}/${dset_part} was successful !"
  done
fi

if [ ${stage} -le 3 ] && [ $stop_stage -ge 3 ]; then
    # Preparing ASR training and validation data;
    log "Parsing the GSS output to lhotse manifests"
    cv_kaldi_manifests_gss=()
    tr_kaldi_manifests=()
    for dset in $gss_dsets; do
      # for each dataset get the name and part (dev or train)
      dset_name="$(cut -d'_' -f1 <<<${dset})"
      dset_part="$(cut -d'_' -f2 <<<${dset})"
      python local/gss2lhotse.py -i ${gss_dump_root}/${dset_name}/${dset_part} \
        -o $manifests_root/gss/${dset_name}/${dset_part}/${dset_name}_${dset_part}_gss

      lhotse kaldi export -p $manifests_root/gss/${dset_name}/${dset_part}/${dset_name}_${dset_part}_gss_recordings.jsonl.gz  \
          $manifests_root/gss/${dset_name}/${dset_part}/${dset_name}_${dset_part}_gss_supervisions.jsonl.gz \
          data/kaldi/${dset}/${dset_part}/gss

      ./utils/utt2spk_to_spk2utt.pl data/kaldi/${dset}/${dset_part}/gss/utt2spk > data/kaldi/${dset}/${dset_part}/gss/spk2utt
      ./utils/fix_data_dir.sh data/kaldi/${dset}/${dset_part}/gss

      if [ $dset_part == train ]; then
        tr_kaldi_manifests+=( "data/kaldi/${dset}/${dset_part}/gss" )
      fi

      if [ $dset_part == dev ]; then
        cv_kaldi_manifests_gss+=( "data/kaldi/${dset}/${dset_part}/gss" )
      fi
    done

    if (( ${#cv_kaldi_manifests_gss[@]} )); then
      ./utils/combine_data.sh data/kaldi/dev_gss_all "${cv_kaldi_manifests_gss[@]}"
      ./utils/fix_data_dir.sh data/kaldi/dev_gss_all
    fi
    # not empty
    # Preparing all the ASR data, dumping to Kaldi manifests and then merging all the data
    # train set
    log "Dumping all lhotse manifests to kaldi manifests and merging everything for training set."
    dset_part=train
    mic=ihm
    for dset in chime6 mixer6; do
      for mic in ihm mdm; do
        #if [ $dset == mixer6 ] && [ $mic == ihm ]; then
        #  continue # not used right now
        #fi
      lhotse kaldi export -p ${manifests_root}/${dset}/${dset_part}/${dset}-${mic}_recordings_${dset_part}.jsonl.gz  ${manifests_root}/${dset}/${dset_part}/${dset}-${mic}_supervisions_${dset_part}.jsonl.gz data/kaldi/${dset}/${dset_part}/${mic}
      ./utils/utt2spk_to_spk2utt.pl data/kaldi/${dset}/${dset_part}/${mic}/utt2spk > data/kaldi/${dset}/${dset_part}/${mic}/spk2utt
      ./utils/fix_data_dir.sh data/kaldi/${dset}/${dset_part}/${mic}
      tr_kaldi_manifests+=( "data/kaldi/$dset/$dset_part/$mic" )
      done
    done

    ./utils/combine_data.sh data/kaldi/train_all "${tr_kaldi_manifests[@]}"
    ./utils/fix_data_dir.sh data/kaldi/train_all

    # dev set ihm, useful for debugging and testing how well ASR performs in best conditions
    log "Dumping all lhotse manifests to kaldi manifests for dev set with close-talk microphones."
    cv_kaldi_manifests_ihm=()
    dset_part=dev
    mic=ihm
    for dset in chime6 dipco; do
      lhotse kaldi export -p ${manifests_root}/${dset}/${dset_part}/${dset}-${mic}_recordings_${dset_part}.jsonl.gz  ${manifests_root}/${dset}/${dset_part}/${dset}-${mic}_supervisions_${dset_part}.jsonl.gz data/kaldi/${dset}/${dset_part}/${mic}
      ./utils/utt2spk_to_spk2utt.pl data/kaldi/${dset}/${dset_part}/${mic}/utt2spk > data/kaldi/${dset}/${dset_part}/${mic}/spk2utt
      ./utils/fix_data_dir.sh data/kaldi/${dset}/${dset_part}/${mic}
      cv_kaldi_manifests_ihm+=( "data/kaldi/$dset/$dset_part/${mic}")
    done
    # shellcheck disable=2043
    ./utils/combine_data.sh data/kaldi/dev_ihm_all "${cv_kaldi_manifests_ihm[@]}"
    ./utils/fix_data_dir.sh data/kaldi/dev_ihm_all
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  asr_train_set=kaldi/train_all
  asr_cv_set=kaldi/dev_gss_all
  # Decoding on dev set because test is blind for now
  asr_tt_set="kaldi/chime6_dev/dev/gss/ kaldi/dipco_dev/dev/gss/ kaldi/mixer6_dev/dev/gss/"
  ./asr.sh \
    --lang en \
    --local_data_opts "--train-set ${asr_train_set}" \
    --stage $asr_stage \
    --ngpu $ngpu \
    --asr_args "--batch_size ${asr_batch_size} --scheduler_conf warmup_steps=${asr_warmup} --max_epoch=${asr_max_epochs} --optim_conf lr=${asr_max_lr}" \
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
    --use_word_lm ${use_word_lm} \
    --word_vocab_size ${word_vocab_size} \
    --train_set "${asr_train_set}" \
    --valid_set "${asr_cv_set}" \
    --test_sets "${asr_tt_set}" \
    --bpe_train_text "data/${asr_train_set}/text" \
    --lm_train_text "data/${asr_train_set}/text" "$@"
fi