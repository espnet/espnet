#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=100

num_nodes=1
num_proc_per_node=8
node_rank=0
master_addr=localhost
master_port=8888

train_registered_specifier=""
train_registered_specifier+="dialogue:part2_gen_v1_realistic dialogue:part2_gen_v1_imaginary "
train_registered_specifier+="dialogue:part3_gen_v1_realistic dialogue:part3_gen_v1_imaginary "
train_registered_specifier+="dialogue:part4_gen_v1_realistic dialogue:part4_gen_v1_imaginary "
train_registered_specifier+="dialogue:airbench_train_v1 "
train_registered_specifier+="dialogue:mmau_train_v1 "
train_registered_specifier+="dialogue:asr_v2_inverse_200k "
train_registered_specifier+="dialogue:audiobench_train_v1 "

valid_registered_specifier="text_to_audio:clotho_test"

# test_registered_specifier="dialogue:generation_test_clean"
# test_registered_specifier="dialogue:generation_test_audiocaps "
# test_registered_specifier="dialogue:generation_mmau"

test_registered_specifier+="dialogue:librispeech_test_clean dialogue:librispeech_test_other "
# test_registered_specifier+="dialogue:mmau_test "
# test_registered_specifier+="dialogue:mmar_test "
# test_registered_specifier+="dialogue:airbench_test "
# test_registered_specifier+="dialogue:audiocaps_qa_test dialogue:cn_college_listen_mcq_test dialogue:dream_tts_mcq_test dialogue:public_sg_speech_qa_test dialogue:wavcaps_qa_test "

train_config=conf/train_stage3_qwen3_base.yaml
resume_path=exp/opuslm_v2_stage2_pretrain_base/checkpoints/step_260000

stats_dir=exp/stats_qwen3
# exp_dir=exp/opuslm_v2_stage3_sft_qwen3_geneneration_v2
# exp_dir=exp/opuslm_v2_stage3_sft_gen_v1
# exp_dir=exp/opuslm_v2_stage3_sft_qwen3_geneneration_v2_speech_only
# exp_dir=exp/opuslm_v2_stage3_sft_qwen3_v2
# exp_dir=exp/opuslm_v2_stage3_sft_all_260ksteps_train_v1
# exp_dir=exp/opuslm_v2_stage3_sft_qwen3_combine_v1
# exp_dir=exp/opuslm_v2_stage3_sft_qwen3_combine_v1_2node
exp_dir=exp/opuslm_v2_stage3_sft_qwen3_combine_v2_3node
mkdir -p ${exp_dir}

inference_config=conf/inference_sft.yaml
# inference_step=267570 #275140 #267570
# inference_step=275000
inference_step=272500
inference_nj=16
inference_workers=3

. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  python ../../../espnet2/speechlm/bin/prepare_length_stats.py \
    --train-registered-specifier "${train_registered_specifier}" \
    --valid-registered-specifier "${valid_registered_specifier}" \
    --train-config ${train_config} \
    --output-dir ${stats_dir} \
    --num-workers 88
fi


# 
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Node rank: ${node_rank} launch"

  mkdir -p ${exp_dir}/logs
  timestamp=$(date +"%Y-%m-%d_%H_%M")
  torchrun \
    --nnodes=${num_nodes} \
    --node_rank=${node_rank} \
    --nproc_per_node=${num_proc_per_node} \
    --master_addr=${master_addr} \
    --master_port=${master_port} \
      ../../../espnet2/speechlm/bin/train.py \
      --train-registered-specifier "${train_registered_specifier}" \
      --valid-registered-specifier "${valid_registered_specifier}" \
      --train-config ${train_config} \
      --stats-dir ${stats_dir} \
      --output-dir ${exp_dir} \
      --resume-path ${resume_path} \
      --save-loader-state \
      --wandb-mode online \
      > ${exp_dir}/logs/train_node${node_rank}_${timestamp}.log 2>&1 
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  inference_tag=$(basename "${inference_config%.*}")

  inference_dir=${exp_dir}/inference/${inference_tag}_step_${inference_step}
  mkdir -p ${inference_dir}

  inference_ckpt=(${exp_dir}/checkpoints/step_${inference_step}/global_step*/mp_rank_00_model_states.pt)
  inference_ckpt=${inference_ckpt[0]}

  echo "Start model inference. Log at ${inference_dir}/logs/inference.*.log"
  for specifier in ${test_registered_specifier}; do
    echo "Test specifier: ${specifier}. Log at: ${inference_dir}/logs/inference_${specifier//:/_}.*.log"
    ${cuda_cmd} --gpu 1 JOB=1:${inference_nj} ${inference_dir}/logs/inference_${specifier//:/_}.JOB.log \
      ../../../espnet2/speechlm/bin/inference.py \
        --rank JOB --world-size ${inference_nj} \
        --train-config ${exp_dir}/train.yaml \
        --inference-config ${inference_config} \
        --model-checkpoint ${inference_ckpt} \
        --output-dir ${inference_dir} \
        --test-registered-specifier "${specifier}" \
        --num-worker ${inference_workers} &
  done; wait
  
fi