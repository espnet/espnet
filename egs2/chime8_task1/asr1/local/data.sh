#!/usr/bin/env bash
set -eou pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0
stage=0
skip_stages=("-1")
nlsyms_file=data/nlsyms.txt
dasr_root=
train_set=
gss_dsets=
manifests_root=
gss_dump=
augm_num_data_reps=4
decode_train="dev"
foreground_snrs="20:10:15:5:0"
background_snrs="20:10:15:5:0"

. utils/parse_options.sh
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

gss_dsets=$(echo $gss_dsets | tr "," " ") # split by commas


if [ $decode_train != "train" ]; then
  # stop after gss
  skip_stages=("1" "2")
fi


if [ ${stage} -le 1 ] && ! [[ " ${skip_stages[*]} " =~ " 1 " ]]; then
  all_tr_manifests=()
  all_tr_manifests_ihm=()
  log "Dumping all lhotse manifests to kaldi manifests and merging everything for training set."
  dset_part=train
  for dset in chime6 dipco notsofar1 mixer6; do # (popcornell): mixer6 should be last here otherwise it fails as i reset dset_part below
    for mic in ihm mdm; do
      lhotse kaldi export ${manifests_root}/${dset}/${dset_part}/${dset}-${mic}_recordings_${dset_part}.jsonl.gz  ${manifests_root}/${dset}/${dset_part}/${dset}-${mic}_supervisions_${dset_part}.jsonl.gz data/kaldi/${dset}/${dset_part}/${mic}
      ./utils/utt2spk_to_spk2utt.pl data/kaldi/${dset}/${dset_part}/${mic}/utt2spk > data/kaldi/${dset}/${dset_part}/${mic}/spk2utt
      ./utils/fix_data_dir.sh data/kaldi/${dset}/${dset_part}/${mic}
      if [ $dset == "mixer6" ]; then
        dset_part=train_call
        lhotse kaldi export ${manifests_root}/${dset}/${dset_part}/${dset}-${mic}_recordings_${dset_part}.jsonl.gz  ${manifests_root}/${dset}/${dset_part}/${dset}-${mic}_supervisions_${dset_part}.jsonl.gz data/kaldi/${dset}/${dset_part}/${mic}
        ./utils/utt2spk_to_spk2utt.pl data/kaldi/${dset}/${dset_part}/${mic}/utt2spk > data/kaldi/${dset}/${dset_part}/${mic}/spk2utt
        ./utils/fix_data_dir.sh data/kaldi/${dset}/${dset_part}/${mic}
        dset_part=train_intv
        lhotse kaldi export ${manifests_root}/${dset}/${dset_part}/${dset}-${mic}_recordings_${dset_part}.jsonl.gz  ${manifests_root}/${dset}/${dset_part}/${dset}-${mic}_supervisions_${dset_part}.jsonl.gz data/kaldi/${dset}/${dset_part}/${mic}
        ./utils/utt2spk_to_spk2utt.pl data/kaldi/${dset}/${dset_part}/${mic}/utt2spk > data/kaldi/${dset}/${dset_part}/${mic}/spk2utt
        ./utils/fix_data_dir.sh data/kaldi/${dset}/${dset_part}/${mic}
      fi


      if [ $mic == ihm ] && [ $dset == chime6 ]; then
        # remove bad sessions from ihm, mdm are fine
        log "Removing possibly bad close-talk microphones from CHiME-6 data."
        utils/copy_data_dir.sh data/kaldi/chime6/train/ihm data/kaldi/chime6/train/ihm_bad_sessions # back up
        grep -v -e "^P11_chime6_S03" -e "^P52_chime6_S19" -e "^P53_chime6_S24" -e "^P54_chime6_S24" data/kaldi/chime6/train/ihm_bad_sessions/text > data/kaldi/chime6/train/ihm/text
        utils/fix_data_dir.sh data/kaldi/chime6/train/ihm
      fi

      if [ $mic == ihm ]; then
        all_tr_manifests_ihm+=( "data/kaldi/$dset/$dset_part/$mic" )
      fi

      all_tr_manifests+=( "data/kaldi/$dset/$dset_part/$mic" )
    done
  done

  # now combine all training data
  ./utils/combine_data.sh data/kaldi/train_all_mdm_ihm "${all_tr_manifests[@]}"
  ./utils/fix_data_dir.sh data/kaldi/train_all_mdm_ihm
  # combine all training ihm data, used for augmentation later
  ./utils/combine_data.sh data/kaldi/train_all_ihm "${all_tr_manifests_ihm[@]}"
  ./utils/fix_data_dir.sh data/kaldi/train_all_ihm
fi


if [ $stage -le 2 ] && ! [[ " ${skip_stages[*]} " =~ " 2 " ]]; then
  log "Augmenting close-talk data with MUSAN and CHiME-6 extracted noises."
  local/extract_noises.py ${dasr_root}/chime6/audio/train ${dasr_root}/chime6/transcriptions/train \
    local/distant_audio_list distant_noises_chime6
  local/make_noise_list.py distant_noises_chime6 > distant_noise_list

  # append also notsofar1 and dipco TODO
  # shellcheck disable=SC2011
  ls ${dasr_root}/notsofar1/audio/train/*U*C* -1 | xargs -n1 basename | sed -e 's/\.wav$//' > local/distant_audio_list_notsofar1
  local/extract_noises.py ${dasr_root}/notsofar1/audio/train ${dasr_root}/notsofar1/transcriptions/train \
    local/distant_audio_list_notsofar1 distant_noises_notsofar1
  local/make_noise_list.py distant_noises_notsofar1 >> distant_noise_list
  # shellcheck disable=SC2011
  ls -1 ${dasr_root}/dipco/audio/train/*U*C* | xargs -n1 basename | sed -e 's/\.wav$//' > local/distant_audio_list_dipco
  local/extract_noises.py ${dasr_root}/dipco/audio/train ${dasr_root}/dipco/transcriptions/train \
    local/distant_audio_list_dipco distant_noises_dipco
  local/make_noise_list.py distant_noises_dipco >> distant_noise_list

  noise_list=distant_noise_list

  if [ ! -d RIRS_NOISES/ ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # This is the config for the system using simulated RIRs and point-source noises
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
  rvb_opts+=(--noise-set-parameters "$noise_list")

  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --prefix "rev" \
    --foreground-snrs $foreground_snrs \
    --background-snrs $background_snrs \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 1 \
    --isotropic-noise-addition-probability 1 \
    --num-replications ${augm_num_data_reps} \
    --max-noises-per-minute 1 \
    --source-sampling-rate 16000 \
    data/kaldi/train_all_ihm data/kaldi/train_all_ihm_rvb

  # combine now with total training data
  ./utils/combine_data.sh data/kaldi/train_all_mdm_ihm_rvb data/kaldi/train_all_mdm_ihm data/kaldi/train_all_ihm_rvb
  ./utils/fix_data_dir.sh data/kaldi/train_all_mdm_ihm_rvb
fi


if [ ${stage} -le 3 ] && ! [[ " ${skip_stages[*]} " =~ " 3 " ]]; then
    # Preparing ASR training and validation data;
    log "Parsing the GSS output to Kaldi manifests"
    cv_kaldi_manifests_gss=()
    tr_kaldi_manifests_gss=() # if gss is used also for training
    for dset in $gss_dsets; do
      # for each dataset get the name and part (dev or train)
      dset_name="$(cut -d'_' -f1 <<<${dset})"
      dset_part="$(cut -d'_' -f2 <<<${dset})"
      python local/gss2lhotse.py -i ${gss_dump}/${dset_name}/${dset_part} \
        -o $manifests_root/gss/${dset_name}/${dset_part}/${dset_name}_${dset_part}_gss

      lhotse kaldi export $manifests_root/gss/${dset_name}/${dset_part}/${dset_name}_${dset_part}_gss_recordings.jsonl.gz  \
          $manifests_root/gss/${dset_name}/${dset_part}/${dset_name}_${dset_part}_gss_supervisions.jsonl.gz \
          data/kaldi/${dset_name}/${dset_part}/gss

      ./utils/utt2spk_to_spk2utt.pl data/kaldi/${dset_name}/${dset_part}/gss/utt2spk > data/kaldi/${dset_name}/${dset_part}/gss/spk2utt
      ./utils/fix_data_dir.sh data/kaldi/${dset_name}/${dset_part}/gss

      if [ $dset_part == train ]; then # skip notsofar
         tr_kaldi_manifests_gss+=( "data/kaldi/${dset_name}/${dset_part}/gss")
      fi

      if [ $dset_part == dev ]; then
         cv_kaldi_manifests_gss+=( "data/kaldi/${dset_name}/${dset_part}/gss")
      fi
    done

    if (( ${#tr_kaldi_manifests_gss[@]} )); then
    # possibly combine with all training data the gss training data
      tr_kaldi_manifests_gss+=( data/kaldi/train_all_mdm_ihm_rvb)
      ./utils/combine_data.sh data/kaldi/train_all_mdm_ihm_rvb_gss  "${tr_kaldi_manifests_gss[@]}"
      ./utils/fix_data_dir.sh data/kaldi/train_all_mdm_ihm_rvb_gss
    fi

    if (( ${#cv_kaldi_manifests_gss[@]} )); then # concatenate all gss data to use for validation
      ./utils/combine_data.sh data/kaldi/dev_all_gss  "${cv_kaldi_manifests_gss[@]}"
      ./utils/fix_data_dir.sh data/kaldi/dev_all_gss
    fi

fi



if [ ${stage} -le 4 ] && ! [[ " ${skip_stages[*]} " =~ " 4 " ]]; then
    log "stage 4: Create non linguistic symbols: ${nlsyms_file}"
    if [ -f "${nlsyms_file}" ]; then
      echo "${nlsyms_file} exists already, SKIPPING (please remove if you want to
      override it) !"
    else
      # (popcornell) || true is needed to avoid exiting when grep returns none
      cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "\[" > ${nlsyms_file} || true
      cat ${nlsyms_file}
    fi
fi


log "ASR data preparation successfully finished. [elapsed=${SECONDS}s]"
