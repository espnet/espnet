#!/usr/bin/env bash
# External VAD models (local/baselines/models.py) on test sets, scored as this recipe's
# models: md-eval, collar 0, whole recordings, the union of all speakers as reference.
# Each model's threshold and centred median are tuned on --tune_set over its grid
# (models.py) and then fixed for every set (local/v2/score_vad.py summarize).
#
#   local/baselines/eval.sh --models "silero webrtc0 marblenet" --python <venv>/bin/python \
#       --test_sets "ami_mh_dev_v1 libricss_eval ..." [--out exp/baselines] [--nj 16]
#
# --python is the interpreter with the packages of local/baselines/requirements.txt
# (local/baselines/install.sh). CPU models run as ${train_cmd} JOB=1:--nj, GPU models
# (marblenet) as one ${cuda_cmd} array of --gpu_nj one-GPU tasks, at most --max_gpu at
# once; scoring as ${train_cmd} tasks. Writes <out>/summary.csv (all models).
set -euo pipefail
models="silero webrtc0 marblenet"
python=python3
test_sets=
out=exp/baselines
nj=16
gpu_nj=4
max_gpu=4
tune_set=ami_mh_dev_v1
model_dir=
. utils/parse_options.sh
. ./path.sh
. ./cmd.sh
[ -n "${test_sets}" ] || { echo "$0: give --test_sets"; exit 1; }
md_eval=$(command -v md-eval.pl) || { echo "$0: md-eval.pl not found (tools/sctk)"; exit 1; }
sets=()
for s in ${test_sets}; do sets+=("dump/raw/${s}"); done
mkdir -p "${out}/log"
md_opt=()
[ -n "${model_dir}" ] && md_opt=(--model_dir "${model_dir}")
for m in ${models}; do
    gpu=$(cd local/baselines && "${python}" -c "from models import MODELS; print(int(MODELS['${m}'].gpu))")
    if [ "${gpu}" = 1 ]; then
        ${cuda_cmd} --max-jobs-run "${max_gpu}" --gpu 1 JOB=1:"${gpu_nj}" "${out}/log/${m}.JOB.log" \
            "${python}" local/baselines/run.py --model "${m}" --out "${out}" \
                --sets "${sets[@]}" --task JOB --ntasks "${gpu_nj}" ${md_opt[@]+"${md_opt[@]}"}
    else
        ${train_cmd} JOB=1:"${nj}" "${out}/log/${m}.JOB.log" \
            "${python}" local/baselines/run.py --model "${m}" --out "${out}" \
                --sets "${sets[@]}" --task JOB --ntasks "${nj}" ${md_opt[@]+"${md_opt[@]}"}
    fi
done
: > "${out}/score_tasks.txt"
dirs=()
for m in ${models}; do
    read -r shift offset ths meds < <(python3 -c "import json; d = json.load(open('${out}/${m}/frames.json')); print(d['shift'], d['offset'], ':'.join(map(str, d['thresholds'])), ':'.join(map(str, d['medians'])))")
    for s in ${test_sets}; do
        d=${out}/${m}/diar_eval/diarized_${s}
        sort "${d}"/diarize.*.scp > "${d}/diarize.scp"
        [ "$(wc -l < "${d}/diarize.scp")" = "$(wc -l < "dump/raw/${s}/wav.scp")" ] ||
            { echo "$0: ${d}/diarize.scp misses recordings"; exit 1; }
        echo "python3 local/v2/score_vad.py score ${d} dump/raw/${s} ${d}/scoring --md_eval ${md_eval}" \
            "--thresholds ${ths//:/ } --medians ${meds//:/ } --shift ${shift} --offset ${offset}" \
            >> "${out}/score_tasks.txt"
    done
    dirs+=("${out}/${m}/diar_eval")
done
n=$(wc -l < "${out}/score_tasks.txt")
${train_cmd} JOB=1:"${n}" "${out}/log/score.JOB.log" local/v2/task_line.sh "${out}/score_tasks.txt" JOB
python3 local/v2/score_vad.py summarize "${out}/summary.csv" "${dirs[@]}" --tune_set "${tune_set}"
