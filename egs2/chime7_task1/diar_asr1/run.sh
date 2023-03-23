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
pyannote_access_token=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 #FIXME #TODO
diarization_dir=exp/diarization

# GSS config
asr_use_pretrained=
asr_decode_only=0

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh


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


if [ $diarization_backend == pyannote ]; then
  # check if pyannote is installed
  if ! python3 -c "import pyannote.audio" &> /dev/null; then
    log "Installing Pyannote Audio."
    (
        python3 -m pip install pyannote-audio
    )
  fi

  # check if doverlap is installed too
  if ! command -v dover-lap &>/dev/null; then
  log "Installing DOVER-Lap."
  (
        python3 -m pip install dover-lap
  )
  fi

  for dset in chime6 dipco mixer6; do
    for split in dev; do
        if [ $dset == mixer6 ]; then
          mic_regex="(CH0[[4-9]|10])" # exclude close-talk CH01, CH02, CH03
        else
          mic_regex="(U[0-9]+)" # exclude close-talk
        fi
        # diarizing with pyannote + ensembling across mics with dover-lap
        python local/pyannote_diarize.py --in-dir ${chime7_root}/${dset}/audio/${split} \
              --uem ${chime7_root}/${dset}/uem/${split}/all.uem --mic_regex $mic_regex \
              --out_folder ${diarization_dir}/${dset}/${split} --token $pyannote_access_token
    done
  done
fi


if [ ${stage} -le 1 ] && [ $stop_stage -ge 1 ]; then
  # parse all datasets to lhotse
  for dset in chime6 dipco mixer6; do
    for dset_part in dev; do
      log "Creating lhotse manifests for ${dset} in $manifests_root/${dset}"
      python local/get_lhotse_manifests.py -c $chime7_root \
           -d $dset \
           -p $dset_part \
           -o $manifests_root --diar_jsons_root "$diarization_dir" \
           --ignore_shorter 0.2
    done
  done
fi


if [ ${stage} -le 3 ] && [ $stop_stage -ge 3 ]; then
  log("Performing GSS+Channel Selection+ASR inference on diarized output")
  # now that we have diarized the dataset, we can run the sub-track 1 baseline
  # and use the diarization output in place of oracle diarization.
  ./../asr1/run.sh --chime7-root $chime7_root --stage 2 --stop-stage 3 --ngpu $ngpu \
        --use-pretrained $asr_use_pretrained \
        --decode_only $asr_decode_only --gss-max-batch-dur $gss_max_batch_dur
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
    # the content of this output folder is what you should send for evaluation to the
    # organizers.
  done
  split=dev
  LOG_OUT=${asr_exp}/${inference_tag}/scoring/scoring.log
  python local/da_wer_scoring.py -s ${asr_exp}/${inference_tag}/chime7dasr_hyp/$split \
     -r $chime7_root -p $split -o ${asr_exp}/${inference_tag}/scoring -d 1 2>&1 | tee $LOG_OUT
fi
