#!/bin/bash

# Copyright 2019 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo "$0 $*"  # Print the command line for logging
. ./path.sh

nj=1
do_delta=false
eval_tts_model=true
db_root=""
backend=pytorch
wer=false
vocoder=
api=v2
help_message="Usage: $0 <asr_model> <outdir> <subset>"

. utils/parse_options.sh

asr_model=$1
outdir=$2
set_name=$3

if [ $# != 3 ]; then
    echo "${help_message}"
    exit 1;
fi

set -euo pipefail

# Decide wavdir
if [ ${vocoder} != "None" ]; then
    # select vocoder type (GL, PWG)
    if [ ${vocoder} == "PWG" ]; then
        wavdir=${outdir}_denorm/${set_name}/pwg_wav
    elif [ ${vocoder} == "GL" ]; then
        wavdir=${outdir}_denorm/${set_name}/wav
    else
        echo "Vocoder type other than GL, PWG is not supported!"
        exit 1
    fi

    name=$set_name
else
    echo "Please specify vocoder."
    exit 1
fi

# Get transcription

echo "step 0: Model preparation"
# ASR model selection for CER objective evaluation 
# Please add new model if you want to use your ASR model.
asr_model_dir="exp/${asr_model}_asr"
case "${asr_model}" in
    "librispeech.transformer.ngpu4") asr_url="https://drive.google.com/open?id=1BtQvAnsFvVi-dp_qsaFP7n4A_5cwnlR6" \
      asr_cmvn="${asr_model_dir}/data/train_960/cmvn.ark" \
      asr_pre_decode_config="${asr_model_dir}/conf/tuning/decode_pytorch_transformer_large.yaml" \
      recog_model="${asr_model_dir}/exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug/results/model.val5.avg.best" \
      lang_model="${asr_model_dir}/exp/irielm.ep11.last5.avg/rnnlm.model.best" \
      asr_dict="data/decode_dict/X.txt" ;;
    
    "aishell.transformer") asr_url="https://drive.google.com/open?id=1BIQBpLRRy3XSMT5IRxnLcgLMirGzu8dg" \
      asr_cmvn="${asr_model_dir}/data/train_sp/cmvn.ark" \
      asr_pre_decode_config="${asr_model_dir}/conf/decode.yaml" \
      recog_model="${asr_model_dir}/exp/train_sp_pytorch_train_pytorch_transformer_lr1.0/results/model.last10.avg.best" \
      new_recog_model="${asr_model_dir}/exp/train_sp_pytorch_train_pytorch_transformer_lr1.0/results/model.last10.avg.best." \
      lang_model="${asr_model_dir}/exp/train_rnnlm_pytorch_lm/rnnlm.model.best" \
      asr_dict="${asr_model_dir}/data/train_sp_units.txt" ;;

    *) echo "No such models: ${asr_model}"; exit 1 ;;
esac

# ASR model download (librispeech)
if [ ! -e ${asr_model_dir}/.complete ]; then
    mkdir -p ${asr_model_dir}
    download_from_google_drive.sh ${asr_url} ${asr_model_dir} ".tar.gz"
    touch ${asr_model_dir}/.complete
fi
echo "ASR model: ${asr_model_dir} exits."

# Change "input_layer" to "embed" in the recog_model if using aishell
# Due to backward compatibility
if [ ${asr_model} = "aishell.transformer" ]; then
    new_recog_model=${recog_model}.new
    if [ ! -e ${new_recog_model} ]; then
        local/ob_eval/change_key_name.py ${recog_model} ${new_recog_model}
    fi
    recog_model=${new_recog_model}
fi

# setting dir
asr_data_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.data"
asr_fbank_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.fbank"
asr_feat_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.dump"
asr_result_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.result"


echo "step 1: Data preparation for ASR"
# Data preparation for ASR
if [ ${asr_model} = "aishell.transformer" ]; then
    local/ob_eval/data_prep_for_asr_mandarin.sh ${wavdir} ${asr_data_dir}/${name} ${db_root}/prompts/Man_transcriptions.txt
elif [ ${asr_model} = "librispeech.transformer.ngpu4" ]; then
    local/ob_eval/data_prep_for_asr_english.sh ${wavdir} ${asr_data_dir}/${name} ${db_root}/prompts/Eng_transcriptions.txt
fi
utils/validate_data_dir.sh --no-feats ${asr_data_dir}/${name}


echo "step 2: Feature extraction for ASR"
# Feature extraction for ASR
steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} \
  --write_utt2num_frames true \
  --write_utt2dur false \
  ${asr_data_dir}/${name} \
  ${outdir}_denorm.ob_eval/${asr_model}_asr.make_fbank/${name} \
  ${asr_fbank_dir}/${name}

utils/fix_data_dir.sh ${asr_data_dir}/${name}

dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
  ${asr_data_dir}/${name}/feats.scp ${asr_cmvn} ${outdir}_denorm.ob_eval/${asr_model}_asr.dump_feats/${name} \
  ${asr_feat_dir}/${name}


echo "step 3: Dictionary and Json Data Preparation for ASR"
# Dictionary and Json Data Preparation for ASR
if [ ${asr_model} = "aishell.transformer" ]; then
    # [dirty] make sure to generate the dict file in aishell recipe
    cp ../../csmsc/tts1/exp/aishell.transformer_asr/data/train_sp_units.txt exp/aishell.transformer_asr/data/
elif [ ${asr_model} = "librispeech.transformer.ngpu4" ]; then
    mkdir -p ${asr_dict%/*}
    echo "<unk> 1" > ${asr_dict}
fi

data2json.sh --feat ${asr_feat_dir}/${name}/feats.scp \
  ${asr_data_dir}/${name} ${asr_dict} > ${asr_feat_dir}/${name}/data.json


echo "step 4: ASR decoding"
# ASR decoding
asr_decode_config="conf/ob_eval/decode_asr.yaml"
cat < ${asr_pre_decode_config} | sed -e 's/beam-size: 60/beam-size: 10/' > ${asr_decode_config}

# split data
splitjson.py --parts ${nj} ${asr_feat_dir}/${name}/data.json

# set batchsize 0 to disable batch decoding    
${decode_cmd} JOB=1:${nj} ${asr_result_dir}.${api}/${name}/log/decode.JOB.log \
    asr_recog.py \
      --config ${asr_decode_config} \
      --ngpu 0 \
      --backend ${backend} \
      --batchsize 0 \
      --recog-json ${asr_feat_dir}/${name}/split${nj}utt/data.JOB.json \
      --result-label ${asr_result_dir}.${api}/${name}/data.JOB.json \
      --model ${recog_model} \
      --api ${api} \
      --rnnlm ${lang_model}

# calculate CER
if [ ${asr_model} = "aishell.transformer" ]; then
    score_sclite.sh --wer ${wer} ${asr_result_dir}.${api}/${name} ${asr_dict}
elif [ ${asr_model} = "librispeech.transformer.ngpu4" ]; then
    score_sclite_wo_dict.sh --wer ${wer} ${asr_result_dir}.${api}/${name}
fi
