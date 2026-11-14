#!/usr/bin/env bash
# Evaluate checkpoints of a training run on the test sets at given look-aheads, through
# diar.sh stage 6 (inference: one array of one-GPU tasks per set, split by length, at most
# --inference_max_jobs_run at once) and stage 7 (local/score.sh). Stage 6 writes
# <diar_exp>/diarized_<set> whatever the model, so every checkpoint and look-ahead gets its
# own exp directory, <out>/<run>_<checkpoint>_la<frames>/diar_eval, linking the run's
# config.yaml and the checkpoint.
#
#   local/eval_vad.sh --src_exp exp/diar_<tag> --test_sets "<set> ..." \
#       [--models "valid.acc.best.pth ..."] [--look_aheads "0 124"] [--out exp/eval_vad]
#   then: python3 local/v2/score_vad.py summarize <csv> <out>/*/diar_eval
set -euo pipefail
src_exp=
models="valid.acc.best.pth valid.acc.ave_2best.pth valid.acc.ave_5best.pth"
look_aheads="0 124"
test_sets=
out=exp/eval_vad
inference_nj=16
inference_max_jobs_run=4
tune_set=ami_mh_dev
. utils/parse_options.sh
[ -n "${src_exp}" ] && [ -n "${test_sets}" ] || { echo "$0: give --src_exp and --test_sets"; exit 1; }
src=$(cd "${src_exp}" && pwd)
read -r first rest <<< "${test_sets}"
for m in ${models}; do
    [ -f "${src}/${m}" ] || { echo "$0: no ${src}/${m}"; exit 1; }
    for la in ${look_aheads}; do
        name=${src##*/}_${m%.pth}_la${la}
        e=${out}/${name}
        mkdir -p "${e}/diar_eval"
        ln -sfn "${src}/config.yaml" "${e}/diar_eval/config.yaml"
        ln -sfn "${src}/${m}" "${e}/diar_eval/${m}"
        # whole recordings, one per batch (conf/decode.yaml), at this look-ahead
        { cat conf/decode.yaml; echo "look_ahead: ${la}"; } > "${e}/decode.yaml"
        echo "== ${name}"
        # stage 6 also runs on --valid_set: the first test set, not the training dev set
        ./run.sh --stage 6 --stop_stage 7 --expdir "${e}" --diar_tag eval \
            --inference_model "${m}" --inference_config "${e}/decode.yaml" \
            --valid_set "${first}" --test_sets "${rest}" --gpu_inference true \
            --inference_nj "${inference_nj}" --inference_max_jobs_run "${inference_max_jobs_run}" \
            --split_by_length true \
            --local_score_opts "--tune_set ${tune_set} --score_valid true"
    done
done
