#!/usr/bin/env bash
# Evaluate checkpoints at given look-aheads on the test sets with one array of GPU tasks and
# one array of CPU tasks, instead of a diar.sh stage-6 run per model, look-ahead and set:
#   1. local/v2/vad_infer.py plan: (model, look-ahead, set, recording) items in --tasks tasks
#      of about equal audio, few models per task
#   2. ${cuda_cmd} --gpu 1 JOB=1:<tasks>, at most --max_jobs at once: vad_infer.py run (each
#      model loaded once per task, recordings by length in batches); outputs where stage 6
#      writes them, <out>/<model>_la<frames>/diar_eval/diarized_<set>
#   3. ${train_cmd} JOB=1:<sets x models x 2>: local/v2/score_vad.py score of every set,
#      centred and causal medians (md-eval, collar 0, whole recordings)
#   4. score_vad.py summarize: the setting with the lowest error on --tune_set per model,
#      <out>/summary.csv (centred median) and summary_causal.csv
#
#   local/eval_vad_array.sh --src_exp exp/diar_<tag> --test_sets "<set> ..." \
#       [--models "valid.acc.best.pth ..."] [--look_aheads "0 124"] [--out exp/eval_vad]
set -euo pipefail
src_exp=
models="valid.acc.best.pth valid.acc.ave_2best.pth valid.acc.ave_5best.pth"
look_aheads="0 124"
test_sets=
out=exp/eval_vad
tasks=12
max_jobs=4
batch_seconds=14400
tune_set=ami_mh_dev
thresholds="0.3 0.4 0.5 0.6 0.7 0.8 0.9"
centred_medians="1 11"
causal_medians="3 5 7 9 11 15 21"
. utils/parse_options.sh
. ./path.sh
. ./cmd.sh
[ -n "${src_exp}" ] && [ -n "${test_sets}" ] || { echo "$0: give --src_exp and --test_sets"; exit 1; }
src=$(cd "${src_exp}" && pwd)
md_eval=$(command -v md-eval.pl) || { echo "$0: md-eval.pl not found (tools/sctk)"; exit 1; }
mkdir -p "${out}/log"
specs=() names=()
for m in ${models}; do
    [ -f "${src}/${m}" ] || { echo "$0: no ${src}/${m}"; exit 1; }
    specs+=(--model "${src##*/}_${m%.pth}=${src}/${m}")
    for la in ${look_aheads}; do names+=("${src##*/}_${m%.pth}_la${la}"); done
done
sets=()
for s in ${test_sets}; do sets+=("dump/raw/${s}"); done
# shellcheck disable=SC2086
python3 local/v2/vad_infer.py plan --out "${out}" --tasks "${tasks}" "${specs[@]}" \
    --look_aheads ${look_aheads} --sets "${sets[@]}"
${cuda_cmd} --max-jobs-run "${max_jobs}" --gpu 1 JOB=1:"${tasks}" "${out}/log/infer.JOB.log" \
    python3 local/v2/vad_infer.py run "${out}/plan/task.JOB.tsv" --batch_seconds "${batch_seconds}"
grep -h "peak GPU memory" "${out}"/log/infer.*.log | sed 's/.*INFO //'
: > "${out}/score_tasks.txt"
for n in "${names[@]}"; do
    for s in ${test_sets}; do
        d=${out}/${n}/diar_eval/diarized_${s}
        sort "${d}"/diarize.*.scp > "${d}/diarize.scp"
        [ "$(wc -l < "${d}/diarize.scp")" = "$(wc -l < "dump/raw/${s}/wav.scp")" ] ||
            { echo "$0: ${d}/diarize.scp misses recordings"; exit 1; }
        echo "python3 local/v2/score_vad.py score ${d} dump/raw/${s} ${d}/scoring" \
            "--md_eval ${md_eval} --thresholds ${thresholds} --medians ${centred_medians}" \
            >> "${out}/score_tasks.txt"
        echo "python3 local/v2/score_vad.py score ${d} dump/raw/${s} ${d}/scoring_causal" \
            "--md_eval ${md_eval} --thresholds ${thresholds} --medians ${causal_medians}" \
            "--median_mode causal" >> "${out}/score_tasks.txt"
    done
done
n=$(wc -l < "${out}/score_tasks.txt")
${train_cmd} JOB=1:"${n}" "${out}/log/score.JOB.log" \
    local/v2/task_line.sh "${out}/score_tasks.txt" JOB
dirs=()
for n in "${names[@]}"; do dirs+=("${out}/${n}/diar_eval"); done
python3 local/v2/score_vad.py summarize "${out}/summary.csv" "${dirs[@]}" --tune_set "${tune_set}"
python3 local/v2/score_vad.py summarize "${out}/summary_causal.csv" "${dirs[@]}" \
    --tune_set "${tune_set}" --scoring scoring_causal
