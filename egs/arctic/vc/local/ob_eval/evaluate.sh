#!/bin/bash

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Evaluation script for VC or ground truth


echo "$0 $*"  # Print the command line for logging
. ./path.sh

nj=1
do_delta=false
eval_model=true
db_root=""
backend=pytorch
api=v2
vocoder=
mcep_dim=24
shift_ms=5
asr_model="librispeech.transformer.ngpu4"
help_message="Usage: $0 <outdir> <subset> <srcspk> <trgspk>"

. utils/parse_options.sh

outdir=$1
name=$2  # `dev` or `eval`
srcspk=$3
trgspk=$4

if [ $# != 4 ]; then
    echo "${help_message}"
    exit 1;
fi

set -euo pipefail

# 
if ${eval_model}; then
    echo "Evaluate: converted speech"

    set_name=${srcspk}_${trgspk}_${name}
    
    # Decide wavdir depending on vocoder
    if [ ! -z ${vocoder} ]; then
        # select vocoder type (GL, PWG)
        if [ ${vocoder} == "PWG" ]; then
            wavdir=${outdir}_denorm/${set_name}/pwg_wav
        elif [ ${vocoder} == "GL" ]; then
            wavdir=${outdir}_denorm/${set_name}/wav
        else
            echo "Vocoder type other than GL, PWG is not supported!"
            exit 1
        fi
    else
        echo "Please specify vocoder."
        exit 1
    fi
else
    echo "Evaluate: ground truth"
    
    set_name=${trgspk}_${name}
    
    expdir=exp/ground_truth
    wavdir=${expdir}/sym_link_denorm/${set_name}/wav
    
    if [ ${db_root} == "" ]; then
        echo "Please set --db_root"; exit 1
    fi
    mkdir -p ${wavdir}
    cat < data/${set_name}/wav.scp | awk '{print $1}' | while read -r f; do
        filename=$(echo "$f" | sed 's/${trgspk}_//g')
        if [ -L ${wavdir}/${filename}.wav ]; then
            unlink ${wavdir}/${filename}.wav
        fi
        ln -s ${db_root}/cmu_us_${trgspk}_arctic/wav/${filename}.wav ${wavdir}/${filename}.wav
    done
fi

echo "MCD calculation"
mcd_file=${outdir}_denorm/${set_name}/mcd.log
minf0=$(awk '{print $1}' conf/${trgspk}.f0)
maxf0=$(awk '{print $2}' conf/${trgspk}.f0)
${decode_cmd} ${mcd_file} \
    local/ob_eval/mcd_calculate.py \
        --wavdir ${wavdir} \
        --gtwavdir ${db_root}/cmu_us_${trgspk}_arctic/wav \
        --mcep_dim ${mcep_dim} \
        --shiftms ${shift_ms} \
        --f0min ${minf0} \
        --f0max ${maxf0}
grep 'Mean' ${mcd_file}

echo "step 0: Model preparation"
# ASR model selection for CER/WER objective evaluation 
asr_model_dir="exp/${asr_model}_asr"
    case "${asr_model}" in
        "librispeech.transformer.ngpu4") asr_url="https://drive.google.com/open?id=1BtQvAnsFvVi-dp_qsaFP7n4A_5cwnlR6" \
            asr_cmvn="${asr_model_dir}/data/train_960/cmvn.ark" \
            asr_pre_decode_config="${asr_model_dir}/conf/tuning/decode_pytorch_transformer_large.yaml" \
            recog_model="${asr_model_dir}/exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug/results/model.val5.avg.best" \
            lang_model="${asr_model_dir}/exp/irielm.ep11.last5.avg/rnnlm.model.best" ;;

    *) echo "No such models: ${asr_model}"; exit 1 ;;
esac

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
cp data/${trgspk}_${name}/text ${asr_data_dir}/${set_name}/text
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
grep 'Mean' ${mcd_file}
