#!/usr/bin/env bash
# Evaluate one checkpoint at one look-ahead on a Kaldi-style test set of another corpus
# (wav.scp with audio file paths, rttm, optional UEM), at a fixed threshold and median,
# with the scorer of local/eval_vad_array.sh. Runs on one GPU (or the CPU), outside a
# recipe dump: e.g. DIHARD III, MSDWild, RAMC on another cluster.
#
#   local/eval_vad_external.sh --data <kaldi dir> --set <name> --model <dir>/<ckpt>.pth \
#       --look_ahead 0 --threshold 0.7 --median 11 --out <dir> [--name <model name>] \
#       [--wav_dir <dir>] [--uem <file>] [--md_eval md-eval.pl] [--batch_seconds 3600]
#
# The model directory holds config.yaml next to the checkpoint. Steps:
#   1. local/v2/external_set.py: <out>/data/<set>: wav.scp (paths, from --wav_dir when
#      the list was written elsewhere), utt2num_samples, rttm (all speakers' segments
#      joined into one pseudo speaker), uem (all.uem or uem of the dir, or --uem)
#   2. local/v2/vad_infer.py plan/run: speech probabilities of every recording
#   3. local/v2/score_vad.py score: md-eval -c 0 at the threshold and median (centred
#      or --median_mode causal); with a UEM only its regions are scored (md-eval -u),
#      without one each recording whole
#   4. local/v2/score_vad.py summarize: <out>/summary.csv, one row (the format of
#      eval_vad_array.sh), model <name>_la<look-ahead>
set -euo pipefail
data=
set_name=
model=
name=
look_ahead=0
threshold=0.7
median=11
median_mode=centred
out=
wav_dir=
uem=
md_eval=md-eval.pl
batch_seconds=3600
while [ $# -gt 0 ]; do
    case "$1" in
        --data) data=$2 ;;
        --set) set_name=$2 ;;
        --model) model=$2 ;;
        --name) name=$2 ;;
        --look_ahead) look_ahead=$2 ;;
        --threshold) threshold=$2 ;;
        --median) median=$2 ;;
        --median_mode) median_mode=$2 ;;
        --out) out=$2 ;;
        --wav_dir) wav_dir=$2 ;;
        --uem) uem=$2 ;;
        --md_eval) md_eval=$2 ;;
        --batch_seconds) batch_seconds=$2 ;;
        *) echo "$0: unknown option $1"; exit 1 ;;
    esac
    shift 2
done
[ -n "${data}" ] && [ -n "${set_name}" ] && [ -n "${model}" ] && [ -n "${out}" ] ||
    { echo "$0: give --data, --set, --model and --out"; exit 1; }
[ -f "${model}" ] && [ -f "$(dirname "${model}")/config.yaml" ] ||
    { echo "$0: no ${model} or its config.yaml"; exit 1; }
name=${name:-$(basename "$(dirname "${model}")")}
v2=$(dirname "$0")/v2
opts=()
[ -n "${wav_dir}" ] && opts+=(--wav_dir "${wav_dir}")
[ -n "${uem}" ] && opts+=(--uem "${uem}")
python3 "${v2}/external_set.py" "${data}" "${out}/data/${set_name}" ${opts[@]+"${opts[@]}"}
python3 "${v2}/vad_infer.py" plan --out "${out}" --tasks 1 --model "${name}=${model}" \
    --look_aheads "${look_ahead}" --sets "${out}/data/${set_name}"
python3 "${v2}/vad_infer.py" run "${out}/plan/task.1.tsv" --batch_seconds "${batch_seconds}"
d=${out}/${name}_la${look_ahead}/diar_eval/diarized_${set_name}
sort "${d}"/diarize.*.scp > "${d}/diarize.scp"
[ "$(wc -l < "${d}/diarize.scp")" = "$(wc -l < "${out}/data/${set_name}/wav.scp")" ] ||
    { echo "$0: ${d}/diarize.scp misses recordings"; exit 1; }
uem_opt=()
[ -s "${out}/data/${set_name}/uem" ] && uem_opt=(--uem "${out}/data/${set_name}/uem")
python3 "${v2}/score_vad.py" score "${d}" "${out}/data/${set_name}" "${d}/scoring" \
    --md_eval "${md_eval}" --thresholds "${threshold}" --medians "${median}" \
    --median_mode "${median_mode}" ${uem_opt[@]+"${uem_opt[@]}"}
python3 "${v2}/score_vad.py" summarize "${out}/summary.csv" \
    "${out}/${name}_la${look_ahead}/diar_eval" --tune_set "${set_name}"
cat "${out}/summary.csv"
