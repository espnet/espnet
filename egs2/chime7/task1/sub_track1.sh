#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
# main stages
stage=1
stop_stage=1


# NOTE, use absolute paths !
chime7_root=${PWD}/chime7_task1
chime5_root= 
chime6_root=/raid/users/popcornell/CHiME6/espnet/egs2/chime6/asr1/CHiME6 # will be created automatically from chime5
# but if you have it already you can use your existing one.
dipco_root=/raid/users/popcornell/CHiME6/DipCO/DiPCo/ # this will be automatically downloaded
mixer6_root=/raid/users/popcornell/mixer6/
manifests_root=./data/lhotse

# dataprep options
cmd_dprep=run.pl
dprep_stage=2
gss_dump_root=./exp/gss
ngpu=4  # set equal to the number of GPUs you have, used for GSS and ASR training

tr_dsets=train
cv_dsets=dev
tt_dsets=

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

# gss config
max_batch_dur=120 # set accordingly to your GPU VRAM, here I used 40GB
nj_gss=6
cmd_gss=run.pl

# asr config
# NOTE: if you get OOM reduce the batch size
#asr_tr_set="chime6_mdm mixer6"
#asr_tr_mics""=mdm ihm gss""
#asr_cv_set=#kaldi/dipco/dev_ihm
asr_stage=0 # starts at 13 for inference only

bpe_nlsyms=""
asr_config=conf/tuning/train_asr_transformer_wavlm_lr1e-4_specaugm_accum1_preenc128_warmup40k.yaml
inference_config="conf/decode_asr_transformer.yaml"
lm_config="conf/train_lm.yaml"
use_lm=false
use_word_lm=false
word_vocab_size=65000
asr_tr_set=kaldi/dipco/dev_ihm
asr_cv_set=kaldi/dipco/dev_ihm

if [ ${stage} -le 0 ] && [ $stop_stage -ge 0 ]; then
  # create the dataset
  ./create_dataset.sh --chime6-root $chime6_root --stage $dprep_stage  --chime7-root $chime7_root \
	  --dipco-root $dipco_root \
	  --mixer6-root $mixer6_root \
	  --stage $dprep_stage \
	  --train_cmd $cmd_dprep
fi


if [ ${stage} -le 1 ] && [ $stop_stage -ge 1 ]; then
  # parse dset to lhotse
  for dset in chime6 dipco mixer6; do
    for dset_part in train dev; do
      if [ $dset == dipco ] && [ $dset_part == train ]; then
          continue # dipco has no train set
      fi
      echo "Creating lhotse manifests for ${dset} in $manifests_root/${dset}"
      python local/data/get_lhotse_manifests.py -c $chime7_root \
           -d $dset \
           -p $dset_part \
           -o $manifests_root \
           --ignore_shorter 0.2
    done
  done
fi


if [ ${stage} -le 2 ] && [ $stop_stage -ge 2 ]; then
  # check if GSS is installed, if not stop, user must manually install it
  if [ ! command -v gss &> /dev/null ];
    then
      echo "GPU-based Guided Source Separation (GSS) could not be found,
      please refer to the README for how to install it. \n
      See also https://github.com/desh2608/gss for more informations."
      exit
  fi
  for dset_part in dev; do
    for dset_name in mixer6; do
      if [ $dset == dipco ] && [ $dset_part == train ]; then
          continue # dipco has no train set
      fi
      echo "Running Guided Source Separation for ${dset_name}/${dset_part}, results will be in ${gss_dump_root}/${dset_name}/${dset_part}"
        ./run_gss.sh --manifests-dir $manifests_root --dset-name $dset_name \
            --dset-part $dset_part \
            --exp_dir $gss_dump_root \
            --cmd $cmd_gss \
            --nj $ngpu \
            --max_batch_dur $max_batch_dur
      echo "Guided Source Separation processing for ${dset_name}/${dset_part} was successful !"
      echo "Parsing the GSS output to lhotse manifests which will be placed in ${manifests_root}/${dset_name}/${dset_part}"
    done
  done
fi

if [ ${stage} -le 3 ] && [ $stop_stage -ge 3 ]; then
    # Preparing ASR training and validation data;
    for dset_part in dev; do
    for dset_name in mixer6; do
      if [ $dset == dipco ] && [ $dset_part == train ]; then
          continue # dipco has no train set
      fi
      python local/data/gss2lhotse.py -i $gss_dump_root -o $manifests_root/gss/
    # parse gss output to kaldi manifests
    # train set
    tr_kaldi_manifests=()
    dset_part=train
    mic=ihm
    for dset in chime6 mixer6; do
      for mic in ihm mdm; do
        if [ $dset == mixer6 ] && [ $mic == ihm ]; then
          continue # not used right now
        fi
      lhotse kaldi export -p ${manifests_root}/${dset}/${dset_part}/${dset}-${mic}_recordings_${dset_part}.jsonl.gz  ${manifests_root}/${dset}/${dset_part}/${dset}-${mic}_supervisions_${dset_part}.jsonl.gz data/kaldi/${dset}/${dset_part}/${mic}
      ./utils/utt2spk_to_spk2utt.pl data/kaldi/${dset}/${dset_part}/${mic}/utt2spk > data/kaldi/${dset}/${dset_part}/${mic}/spk2utt
      ./utils/fix_data_dir.sh data/kaldi/${dset}/${dset_part}/${mic}
      tr_kaldi_manifests+=( "data/kaldi/$dset/$dset_part/$mic" )
      done
    done
    echo ${tr_kaldi_manifests[@]}
    ./utils/combine_data.sh data/kaldi/train_all ${tr_kaldi_manifests[@]}
    ./utils/fix_data_dir.sh data/kaldi/train_all

    # dev set ihm
    cv_kaldi_manifests_ihm=()
    dset_part=dev
    mic=ihm
    for dset in chime6 dipco; do
      lhotse kaldi export -p ${manifests_root}/${dset}/${dset_part}/${dset}-${mic}_recordings_${dset_part}.jsonl.gz  ${manifests_root}/${dset}/${dset_part}/${dset}-${mic}_supervisions_${dset_part}.jsonl.gz data/kaldi/${dset}/${dset_part}/${mic}
      ./utils/utt2spk_to_spk2utt.pl data/kaldi/${dset}/${dset_part}/${mic}/utt2spk > data/kaldi/${dset}/${dset_part}/${mic}/spk2utt
      ./utils/fix_data_dir.sh data/kaldi/${dset}/${dset_part}/${mic}
      cv_kaldi_manifests_ihm+=( "data/kaldi/$dset/$dset_part/$mic" )
    done
    echo ${cv_kaldi_manifests_ihm[@]}
    ./utils/combine_data.sh data/kaldi/dev_ihm_all ${cv_kaldi_manifests_ihm[@]}
    ./utils/fix_data_dir.sh data/kaldi/dev_ihm_all

    # dev set gss
    #dset_part=dev
    #mic=gss
    #for dset in chime6 dipco mixer6; do
     # lhotse kaldi export -p $manifests_root/$dset/$dset_part/$dset- data/kaldi/$dset/$dset_part/$mic
     # $cv_kaldi_manifests_gss+=" data/kaldi/$dset/$dset_part/$mic"
    #done
    #./utils/combine_data.sh data/kaldi/dev_gss $cv_kaldi_manifests_gss
fi


if [ ${stage} -le 4 ] && [ $stop_stage -ge 4 ]; then
  # create dummy non-linguistic symbols file, these are already removed in data prep
  asr_train_set=kaldi/train_all
  asr_cv_set=kaldi/dev_gss_all
  asr_tt_set="kaldi/chime6/dev_gss kaldi/dipco/dev_gss kaldi/mixer6/dev_gss"
  ./asr.sh \
    --lang en \
    --local_data_opts "--train-set ${asr_train_set}" \
    --stage $asr_stage \
    --ngpu $ngpu \
    --token_type bpe \
    --nbpe 500 \
    --bpe_nlsyms "${bpe_nlsyms}" \
    --nlsyms_txt "data/nlsyms.txt" \
    --feats_type raw \
    --audio_format "flac" \
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
