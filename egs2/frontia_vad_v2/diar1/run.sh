#!/usr/bin/env bash
# frontia_vad_v2: voice activity detection trained as single-output diarization
# (diar.sh --num_spk 1) on many domains, balanced per epoch by
# --chunk_upsampling_factor (conf/train.yaml) over the domains in utt2category.
#
#   ./run.sh                    data preparation (local/data.sh), then diar.sh
#   ./run.sh --stage 4          training and evaluation only
#   CMD_BACKEND=pbs ./run.sh --stage 4 --stop_stage 4 --nj 512 --split_by_length true \
#       --stats_args "--batch_size 1 --num_workers 28"
#                               statistics on a cluster: many small tasks (cmd.sh,
#                               conf/pbs.conf with a cap on running tasks; README)
#   CMD_BACKEND=pbs local/data.sh --stage 3 --stop_stage 4 --prep_tasks 256 --align_tasks 64
#                               data preparation on a cluster (corpus and GPU tasks)
#   CMD_BACKEND=pbs ./run.sh --stage 6 --inference_nj 64 --split_by_length true --diar_tag ...
#                               inference and scoring of the test sets on a cluster
#                               (at most 8 GPU tasks at once: --inference_max_jobs_run 8)
#   ./run.sh --stage 5 --stop_stage 5 --ngpu 4 \
#       --diar_config conf/tuning/train_dualmode_v2_lr1e-3.yaml \
#       --diar_tag train_dualmode_v2_lr1e-3     the v2 dual-mode model on 4 GPUs
#
# local/data.sh writes dump/raw/{train,dev} itself, with recordings that have no
# speech (non-speech domains); diar.sh stage 3 would drop those (it rebuilds the
# sets from Kaldi segments), so diar.sh runs stage 2 for the test sets only and
# continues from stage 4.
set -e
set -u
set -o pipefail

stage=1
stop_stage=100
# --stage and --stop_stage are read here; every other option goes to diar.sh
diar_extra=()
while [ $# -gt 0 ]; do
    case "$1" in
        --stage) stage=$2; shift 2 ;;
        --stop_stage) stop_stage=$2; shift 2 ;;
        *) diar_extra+=("$1"); shift ;;
    esac
done

# test sets of frontia_vad_v1 (local/data.sh stage 1)
v1_test="ami,chime6,notsofar,icsi,dipco,ami_mh,alimeeting,aishell4,voxconverse,earnings21,ava_speech,ten_vad"
eval_split_only="aishell4 earnings21 ava_speech ten_vad"  # no dev split
test_sets=""
for c in ${v1_test//,/ }; do
    [[ " ${eval_split_only} " == *" ${c} "* ]] || test_sets+="${c}_dev "
    test_sets+="${c}_eval "
done

diar_opts=(
    --fs 16k
    --audio_format omniio
    --num_spk 1
    --collar 0.0
    --frame_shift 128
    --stats_args "--batch_size 1 --num_workers 2"
    --train_set train
    --valid_set dev
    --test_sets "${test_sets}"
    --diar_config conf/train.yaml
    --diar_tag train
    --inference_config conf/decode.yaml
    --inference_max_jobs_run 8   # GPU inference tasks at once (also cuda_cmd's cap in cmd.sh)
    --skip_dumped true
)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    local/data.sh
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # dumps the test sets; train and dev are marked done by local/data.sh
    ./diar.sh "${diar_opts[@]}" --stage 2 --stop_stage 2
fi
if [ ${stop_stage} -ge 4 ]; then
    # statistics (4), training (5), inference and scoring (6-8) of diar.sh
    ./diar.sh "${diar_opts[@]}" --stage "$(( stage > 4 ? stage : 4 ))" \
        --stop_stage "${stop_stage}" ${diar_extra[@]+"${diar_extra[@]}"}
fi
