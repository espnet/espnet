#!/usr/bin/env bash
#SBATCH --job-name=stages
#SBATCH --partition=RM-shared
#SBATCH --account=cis210027p
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=3-00:00:00
#SBATCH --output=/ocean/projects/cis210027p/mliang4/dailytalk_tts/espnet/egs2/voxtlm_v1/lm1/log/myrun.%j.out
#SBATCH --error=/ocean/projects/cis210027p/mliang4/dailytalk_tts/espnet/egs2/voxtlm_v1/lm1/log/myrun.%j.err

set -e
set -u
set -o pipefail

cd /ocean/projects/cis210027p/mliang4/dailytalk_tts/espnet/egs2/voxtlm_v1/lm1

. ./path.sh
. ./cmd.sh

#--partition=GPU-shared
#--gres=gpu:v100-32:1
# ./local/data_librilight.sh data/librilight
# ./run.sh --stage 1 --stop_stage 1


# cd /ocean/projects/cis210027p/mliang4/dailytalk_tts/espnet/egs2/voxtlm_v1/lm1
# . ./path.sh; . ./cmd.sh; . ./db.sh

# data_dir=data/librilight/speechlm
# data_dir_librispeech_asr=data/librispeech/asr

# # 合并 train（原来失败的那一步，现在脚本已修好，这里用同样的命令手动执行）
# utils/combine_data.sh ${data_dir}/train ${data_dir}/librilight_small ${data_dir}/librilight_medium ${data_dir}/librilight_large

# # 从 Librispeech ASR 拷贝 dev/test（后面会被覆盖，但原脚本就是这个顺序）
# mkdir -p ${data_dir}/dev
# utils/copy_data_dir.sh ${data_dir_librispeech_asr}/dev ${data_dir}/dev
# rm ${data_dir}/dev/text

# mkdir -p ${data_dir}/test
# utils/copy_data_dir.sh ${data_dir_librispeech_asr}/test ${data_dir}/test
# rm ${data_dir}/test/text

# # 为 librilight 准备 Librispeech dev/test 子集（无文本，用作 speechlm 的 dev/test）
# for part in dev-clean dev-other test-clean test-other; do
#     local/librilight/data_prep_librispeech.sh ${LIBRISPEECH}/LibriSpeech/${part} ${data_dir}/${part//-/_}
# done

# # 最终合并 dev / test
# utils/combine_data.sh ${data_dir}/dev ${data_dir}/dev_clean ${data_dir}/dev_other
# utils/combine_data.sh ${data_dir}/test ${data_dir}/test_clean ${data_dir}/test_other

# sed -i 's/\r$//' data/vctk/tts/train/text
# ./run.sh --stage 1 --stop_stage 1 --local_data_opts "--stage 2"
# ./run.sh --stage 2 --stop_stage 2
# ./run.sh --stage 3 --stop_stage 3 --kmeans_opts "--num_threads 64"
./run.sh --stage 3 --stop_stage 3

# ./run.sh --stage 3 --stop_stage 3 --nj 1