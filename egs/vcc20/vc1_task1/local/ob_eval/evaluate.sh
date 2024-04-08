#!/usr/bin/env bash

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Evaluation script for VC or ground truth


echo "$0 $*"  # Print the command line for logging
. ./path.sh

nj=1
do_delta=false
db_root=""
backend=pytorch
api=v2
asr_model_dir=
asr_model="librispeech.transformer.ngpu4"
help_message="Usage: $0 <outdir> <subset> <srcspk> <trgspk> <wavdir>"

. utils/parse_options.sh

outdir=$1
set_name=$2  # `dev` or `eval`
srcspk=$3
trgspk=$4
wavdir=$5

if [ $# != 5 ]; then
    echo "${help_message}"
    exit 1;
fi

set -euo pipefail

echo "step 0: Model preparation"
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
asr_data_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.data"
asr_fbank_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.fbank"
asr_feat_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.dump"
asr_result_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.result"


echo "step 1: Data preparation for ASR"
# Data preparation for ASR
local/ob_eval/data_prep_for_asr.sh ${wavdir} ${asr_data_dir}/${set_name} ${trgspk}
tail -n 25 ${db_root}/prompts/Eng_transcriptions.txt > ${asr_data_dir}/${set_name}/text
sed -i "s/^/${srcspk}_E/" ${asr_data_dir}/${set_name}/text
utils/validate_data_dir.sh --no-feats ${asr_data_dir}/${set_name}


echo "step 2: Feature extraction for ASR"
# Feature extraction for ASR
steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} \
  --write_utt2num_frames true \
  --write_utt2dur false \
  ${asr_data_dir}/${set_name} \
  ${outdir}_denorm.ob_eval/${asr_model}_asr.make_fbank/${set_name} \
  ${asr_fbank_dir}/${set_name}

utils/fix_data_dir.sh ${asr_data_dir}/${set_name}

dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
  ${asr_data_dir}/${set_name}/feats.scp ${asr_cmvn} ${outdir}_denorm.ob_eval/${asr_model}_asr.dump_feats/${set_name} \
  ${asr_feat_dir}/${set_name}


echo "step 3: Dictionary and Json Data Preparation for ASR"
# Dictionary and Json Data Preparation for ASR
asr_dict="data/asr_dict/X.txt"; mkdir -p ${asr_dict%/*}
echo "<unk> 1" > ${asr_dict}

data2json.sh --feat ${asr_feat_dir}/${set_name}/feats.scp \
  ${asr_data_dir}/${set_name} ${asr_dict} > ${asr_feat_dir}/${set_name}/data.json


echo "step 4: ASR decoding"
# ASR decoding
asr_decode_config="conf/ob_eval/decode_asr.yaml"
cat < ${asr_pre_decode_config} | sed -e 's/beam-size: 60/beam-size: 10/' > ${asr_decode_config}

# split data
splitjson.py --parts ${nj} ${asr_feat_dir}/${set_name}/data.json

# set batchsize 0 to disable batch decoding
${decode_cmd} JOB=1:${nj} ${asr_result_dir}.${api}/${set_name}/log/decode.JOB.log \
    asr_recog.py \
      --config ${asr_decode_config} \
      --ngpu 0 \
      --backend ${backend} \
      --batchsize 0 \
      --recog-json ${asr_feat_dir}/${set_name}/split${nj}utt/data.JOB.json \
      --result-label ${asr_result_dir}.${api}/${set_name}/data.JOB.json \
      --model ${recog_model} \
      --api ${api} \
      --rnnlm ${lang_model}

# calculate CER
score_sclite_wo_dict.sh --wer true ${asr_result_dir}.${api}/${set_name}
