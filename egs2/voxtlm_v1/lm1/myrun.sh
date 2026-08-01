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

# ./local/data_librilight.sh data/librilight
./run.sh --stage 1 --stop_stage 1
# ./run.sh --stage 1 --stop_stage 1 --local_data_opts "--stage 2"
# ./run.sh --stage 2 --stop_stage 2
# ./run.sh --stage 3 --stop_stage 3 --nj 1
