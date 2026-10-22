#!/usr/bin/env bash
set -e
set -u
set -o pipefail

setup=voxlingua107
stage=1
stop_stage=10000
local_data_opts=
dumpdir=dump
feats_type=raw
fs=16k
speed_perturb_factors=
python=python3
train_set=
skip_train=false

# Match parse_options.sh: shell configs precede explicit command-line options.
for ((argpos=1; argpos<$#; argpos++)); do
    if [ "${!argpos}" = --config ]; then
        next_arg=$((argpos + 1))
        if [ ! -r "${!next_arg}" ]; then
            echo "$0: missing config '${!next_arg}'" >&2
            exit 2
        fi
        . "${!next_arg}"
    fi
done

# Consume the recipe selector and inspect only the shared options needed below.
# All other options are passed unchanged to the shared LID pipeline.
lid_opts=()
while [ $# -gt 0 ]; do
    option=${1//-/_}
    case "$1" in
        --help|-h)
            echo "Usage: $0 [--setup voxlingua107|combined] [lid.sh options]"
            echo "Combined preparation runs in stages 1-3; resume training at stage 4."
            exec ./lid.sh --help
            ;;
        --*=*|[!\-]*) echo "$0: expected --option value, got '$1'" >&2; exit 2 ;;
    esac
    if [ $# -lt 2 ]; then
        echo "$0: missing value for $1" >&2
        exit 2
    fi
    case "${option}" in
        __setup) setup=$2 ;;
        *)
            lid_opts+=("$1" "$2")
            case "${option}" in
                __stage|__stop_stage|__local_data_opts|__dumpdir|__feats_type|__fs|__speed_perturb_factors|__python|__train_set|__skip_train)
                    printf -v "${option#__}" '%s' "$2"
                    ;;
            esac
            ;;
    esac
    shift 2
done

case "${setup}" in
    voxlingua107)
        default_train_set=train_voxlingua107_lang
        valid_set=${valid_set-dev_voxlingua107_lang}
        test_sets=${test_sets-dev_voxlingua107_lang}
        tsne_set=${tsne_set-dev_voxlingua107_lang}
        lid_config=${lid_config-conf/train.yaml}
        expdir=${expdir-exp_voxlingua107_only}
        ;;
    combined)
        if [ "${feats_type}" != raw ] || [[ "${fs}" != 16k && "${fs}" != 16000 ]] || [ -n "${speed_perturb_factors}" ]; then
            echo "$0: combined preparation requires --feats_type raw, --fs 16k (or 16000), and no speed perturbation." >&2
            exit 2
        fi
        default_train_set=train_all_no_filter_lang
        valid_set=${valid_set-dev_ml_superb2_lang}
        test_sets=${test_sets-"dev_voxlingua107_lang test_voxpopuli_lang test_fleurs_lang dev_ml_superb2_lang dev_dialect_ml_superb2_lang dev_babel_over_10s_lang"}
        tsne_set=${tsne_set-${train_set:-${default_train_set}}}
        lid_config=${lid_config-conf/tuning/combined.yaml}
        expdir=${expdir-exp_combined}
        ;;
    *) echo "$0: unknown setup '${setup}' (choose voxlingua107 or combined)" >&2; exit 2 ;;
esac
train_set=${train_set:-${default_train_set}}

common_opts=(
    --feats_type "${feats_type}"
    --train_set "${train_set}"
    --valid_set "${valid_set}"
    --test_sets "${test_sets}"
    --tsne_set "${tsne_set}"
    --inference_model valid.accuracy.best.pth
    --inference_batch_size 1
    --extract_embd true
    --checkpoint_interval 1000
    --nj 8
    --ngpu 1
    --expdir "${expdir}"
    --lid_config "${lid_config}"
    --max_utt_per_lang_for_tsne 100
)

if [ "${setup}" = combined ] && [ "${stage}" -le 3 ]; then
    prep_stop_stage=${stop_stage}
    if [ "${prep_stop_stage}" -gt 3 ]; then
        prep_stop_stage=3
    fi
    for dataset in voxlingua107 fleurs ml_superb2 babel voxpopuli; do
        case "${dataset}" in
            fleurs|voxpopuli) corpus_test_sets="test_${dataset}_lang" ;;
            ml_superb2) corpus_test_sets=dev_dialect_ml_superb2_lang ;;
            *) corpus_test_sets= ;;
        esac
        ./lid.sh "${common_opts[@]}" "${lid_opts[@]}" \
            --stage "${stage}" --stop_stage "${prep_stop_stage}" \
            --train_set "train_${dataset}_lang" \
            --valid_set "dev_${dataset}_lang" \
            --test_sets "${corpus_test_sets}" --eval_valid_set true \
            --local_data_opts "${local_data_opts} --datasets ${dataset}"
    done
    if [ "${stop_stage}" -ge 3 ]; then
        # Like lid.sh, --skip_data_prep skips stages 1/2, but still runs stage 3.
        babel_splits=dev_babel_lang
        if [ "${skip_train}" = false ]; then
            babel_splits="train_babel_lang ${babel_splits}"
        fi
        bash local/filter_babel_over_10s.sh \
            --dump_dir "${dumpdir}/${feats_type}" --splits "${babel_splits}" \
            --python "${python}"
        if [ "${skip_train}" = false ]; then
            bash local/combine.sh --dump_dir "${dumpdir}/${feats_type}" \
                --train_out "${dumpdir}/${feats_type}/${train_set}" \
                --python "${python}"
        fi
    fi
    if [ "${stop_stage}" -le 3 ]; then
        exit 0
    fi
    stage=4
fi

./lid.sh "${common_opts[@]}" "${lid_opts[@]}" \
    --stage "${stage}" --stop_stage "${stop_stage}"
