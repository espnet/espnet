#!/usr/bin/env bash

# Copyright 2020 Nagoye University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo "$0 $*"  # Print the command line for logging
. ./path.sh

nj=1
db_root=""
backend=pytorch
wer=true
api=v2
help_message="Usage: $0 <asr_model_dir> <expdir> <wavdir> <list> <transcription>"

. utils/parse_options.sh

asr_model_dir=$1
expdir=$2
wavdir=$3
spk=$4
list=$5
transcription=$6

if [ $# -lt 5 ]; then
    echo "${help_message}"
    exit 1;
fi

set -euo pipefail

# We use the pretrained Transformer-ASR model trained on LibriSpeech.
# https://github.com/espnet/espnet/blob/master/egs/librispeech/asr1/RESULTS.md#pytorch-large-transformer-with-specaug-4-gpus--large-lstm-lm
echo "step 0: Model preparation"
asr_url="https://drive.google.com/open?id=1BtQvAnsFvVi-dp_qsaFP7n4A_5cwnlR6"
asr_cmvn="${asr_model_dir}/data/train_960/cmvn.ark"
asr_pre_decode_config="${asr_model_dir}/conf/tuning/decode_pytorch_transformer_large.yaml"
recog_model="${asr_model_dir}/exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug/results/model.val5.avg.best"
lang_model="${asr_model_dir}/exp/irielm.ep11.last5.avg/rnnlm.model.best"

# ASR model download (librispeech)
if [ ! -e ${asr_model_dir}/.complete ]; then
    mkdir -p ${asr_model_dir}
    download_from_google_drive.sh ${asr_url} ${asr_model_dir} ".tar.gz"
    touch ${asr_model_dir}/.complete
fi
echo "ASR model: ${asr_model_dir} exits."

# setting dir
asr_data_dir="${expdir}/data_asr"
asr_fbank_dir="${expdir}/fbank"
asr_feat_dir="${expdir}/dump"
asr_result_dir="${expdir}/result"
asr_dict="${asr_data_dir}/decode_dict/X.txt"

echo "step 1: Data preparation for ASR"
# Data preparation for ASR
local/data_prep_for_asr_english.sh ${wavdir} ${asr_data_dir} ${spk} ${list} ${transcription}
utils/validate_data_dir.sh --no-text --no-feats ${asr_data_dir}

echo "step 2: Feature extraction for ASR"
# Feature extraction for ASR
steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} \
    --write_utt2num_frames true \
    --write_utt2dur false \
    ${asr_data_dir} \
    ${expdir}/make_fbank \
    ${asr_fbank_dir}

# fix_data_dir only works when text exists
if [ ! -z ${transcription} ]; then
    utils/fix_data_dir.sh ${asr_data_dir}
fi

# dump features
dump.sh --cmd "$train_cmd" --nj ${nj} \
  ${asr_data_dir}/feats.scp ${asr_cmvn} ${expdir}/dump_feats/ \
  ${asr_feat_dir}


echo "step 3: Dictionary and Json Data Preparation for ASR"
# Dictionary and Json Data Preparation for ASR
mkdir -p ${asr_dict%/*}
echo "<unk> 1" > ${asr_dict}

data2json.sh --feat ${asr_feat_dir}/feats.scp \
  ${asr_data_dir} ${asr_dict} > ${asr_feat_dir}/data.json


echo "step 4: ASR decoding"
# ASR decoding. We decrease beam-size from 60 to 10 to speed up decoding.
asr_decode_config="conf/decode_asr.yaml"
cat < ${asr_pre_decode_config} | sed -e 's/beam-size: 60/beam-size: 10/' > ${asr_decode_config}

# split data
splitjson.py --parts ${nj} ${asr_feat_dir}/data.json

# set batchsize 0 to disable batch decoding
${decode_cmd} JOB=1:${nj} ${asr_result_dir}/log/decode.JOB.log \
    asr_recog.py \
      --config ${asr_decode_config} \
      --ngpu 0 \
      --backend ${backend} \
      --batchsize 0 \
      --recog-json ${asr_feat_dir}/split${nj}utt/data.JOB.json \
      --result-label ${asr_result_dir}/data.JOB.json \
      --model ${recog_model} \
      --api ${api} \
      --rnnlm ${lang_model}

# calculate CER if ground truth transcription available
# (the script will parse the recognition automatically)
# if not, parse the recognition result.
if [ ! -z ${transcription} ]; then
    score_sclite_wo_dict.sh --wer ${wer} ${asr_result_dir}
else
    concatjson.py ${asr_result_dir}/data.*.json > ${asr_result_dir}/data.json
    json2trn_wo_dict.py ${asr_result_dir}/data.json \
        --refs ${asr_result_dir}/ref_org.wrd.trn \
        --hyps ${asr_result_dir}/hyp_org.wrd.trn
    cat < ${asr_result_dir}/hyp_org.wrd.trn | sed -e 's/▁//' | sed -e 's/▁/ /g' > ${asr_result_dir}/hyp.wrd.trn
fi
