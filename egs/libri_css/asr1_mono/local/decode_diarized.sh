#!/usr/bin/env bash
# Copyright   2019   Ashish Arora, Vimal Manohar
# Copyright   2020   University of Stuttgart (Author: Pavel Denisov)
# Apache 2.0.
# This script takes an rttm file, and performs decoding on on a test directory.
# The output directory contains a text file which can be used for scoring.

# Begin configuration section.
decode_nj=40
stage=0
test_sets=

# ESPnet related variables
dumpdir=dump
do_delta=false
decode_config=conf/decode.yaml
recog_model=model.val5.avg.best
expdir=librispeech.transformer.v1/exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug
lang_model=rnnlm.model.best
lmexpdir=exp/train_rnnlm_pytorch_lm_transformer_cosine_batchsize32_lr1e-4_layer16_unigram5000_ngpu4
nbpe=5000
bpemode=unigram
train_set=train_960
dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}

# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh

set -e # exit on error


if [ $# != 3 ]; then
  echo "Usage: $0 <rttm-dir> <in-data-dir> <out-dir>"
  echo "e.g.: $0 data/rttm data/dev data/dev_diarized"
  exit 1;
fi

rttm_dir=$1
data_in=$2
out_dir=$3

out_set=$(basename $out_dir)

for f in $rttm_dir/rttm $data_in/wav.scp $data_in/text.bak; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

if [ $stage -le 0 ]; then
  echo "$0 copying data files in output directory"
  rm -rf ${out_dir}
  mkdir ${out_dir}
  cp ${data_in}/{wav.scp,utt2spk,utt2spk.bak} ${out_dir}
  utils/data/get_reco2dur.sh ${out_dir}
fi

if [ $stage -le 1 ]; then
  echo "$0 creating segments file from rttm and utt2spk, reco2file_and_channel "
  local/convert_rttm_to_utt2spk_and_segments.py --append-reco-id-to-spkr=true ${rttm_dir}/rttm \
    <(awk '{print $2" "$2" "$3}' ${rttm_dir}/rttm |sort -u) \
    ${out_dir}/utt2spk ${out_dir}/segments

  utils/utt2spk_to_spk2utt.pl ${out_dir}/utt2spk > ${out_dir}/spk2utt
  utils/fix_data_dir.sh ${out_dir} || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0 extracting fbank freatures using segments file"
  fbankdir=fbank
  steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${decode_nj} --write_utt2num_frames true \
      ${out_dir} exp/make_fbank/${out_set} ${fbankdir}
  utils/fix_data_dir.sh ${out_dir}

  feat_dir=${dumpdir}/${out_set}/delta${do_delta}; mkdir -p ${feat_dir}
  awk '{print $1" automatic segments have no transcription"}' ${out_dir}/utt2spk > ${out_dir}/text
  dump.sh --cmd "$train_cmd" --nj ${decode_nj} --do_delta ${do_delta} \
      ${out_dir}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${out_set} \
      ${feat_dir}
  data2json.sh --feat ${feat_dir}/feats.scp --bpecode ${bpemodel}.model \
      ${out_dir} ${dict} > ${feat_dir}/data_${bpemode}${nbpe}.json
  cp ${data_in}/text.bak ${out_dir}/text
fi

if [ $stage -le 3 ]; then
  echo "$0 performing decoding on the extracted features"
  decode_dir=decode_${out_set}_${recog_model}_$(basename ${decode_config%.*})
  feat_dir=${dumpdir}/${out_set}/delta${do_delta}
  # split data
  splitjson.py --parts ${decode_nj} ${feat_dir}/data_${bpemode}${nbpe}.json

  # set batchsize 0 to disable batch decoding
  ${decode_cmd} JOB=1:${decode_nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
    asr_recog.py \
      --config ${decode_config} \
      --ngpu 0 \
      --backend pytorch \
      --batchsize 0 \
      --recog-json ${feat_dir}/split${decode_nj}utt/data_${bpemode}${nbpe}.JOB.json \
      --result-label ${expdir}/${decode_dir}/data.JOB.json \
      --model ${expdir}/results/${recog_model}  \
      --rnnlm ${lmexpdir}/${lang_model} \
      --api v2 \
      --beam-size 30

  # next command is just for json to text conversion
  score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict} >/dev/null
fi

