#!/usr/bin/env bash
# Voice activity detection trained as single-output diarization (diar.sh --num_spk 1).
#
#   stages 1-4  diar.sh: data preparation, Omni-IO dump, short-data removal, stats
#   stage 5     local/train_windowed.sh: training on 10 s windows of the dumped audio
#   stages 6-9  diar.sh: inference on whole recordings, md-eval scoring, packing, upload
#
# The default is the causal model (conf/train.yaml, look-ahead 32 ms). The dual-mode
# model (online and offline with one set of weights):
#   ./run.sh --train_config conf/tuning/train_dualmode.yaml --diar_tag train_dualmode
# scores it online (look-ahead 0). diar.sh writes inference into exp/diar_<diar_tag>,
# so the offline scores (look-ahead 0.99 s) go to a second directory with the same model:
#   mkdir -p exp/diar_train_dualmode_offline
#   ln -s ../diar_train_dualmode/{config.yaml,valid.acc.best.pth} exp/diar_train_dualmode_offline/
#   ./run.sh --stage 6 --diar_tag train_dualmode_offline \
#       --inference_config conf/tuning/decode_offline.yaml
set -e
set -u
set -o pipefail

stage=1
stop_stage=7
train_config=conf/train.yaml
inference_config=conf/decode.yaml
diar_tag=train
inference_model=valid.acc.best.pth
ngpu=1
nj=32
inference_nj=32
gpu_inference=false
inference_max_jobs_run=  # run.pl allows one job per GPU with --gpu: raise it for this small model

. utils/parse_options.sh

# training corpora, then evaluation-only ones (see local/data.sh)
corpora="ami,chime6,notsofar,icsi,dipco,ami_mh,alimeeting,aishell4,voxconverse,earnings21,ava_speech,ten_vad"
eval_split_only="aishell4 earnings21 ava_speech ten_vad"  # no dev split

train_set=train
valid_set=dev
test_sets=""
for c in ${corpora//,/ }; do
    [[ " ${eval_split_only} " == *" ${c} "* ]] || test_sets+="${c}_dev "
    test_sets+="${c}_eval "
done

diar_opts=(
    --fs 16k
    --audio_format omniio
    --num_spk 1
    --collar 0.0
    --frame_shift 128
    --train_set "${train_set}"
    --valid_set "${valid_set}"
    --test_sets "${test_sets}"
    --diar_config "${train_config}"
    --inference_config "${inference_config}"
    --local_data_opts "--corpora ${corpora}"
    --skip_dumped true
    --nj "${nj}"
    --inference_nj "${inference_nj}"
    --gpu_inference "${gpu_inference}"
)
if [ -n "${inference_max_jobs_run}" ]; then
    diar_opts+=(--inference_max_jobs_run "${inference_max_jobs_run}")
fi

if [ "${stage}" -le 4 ]; then
    # stats over whole recordings: one recording per batch
    ./diar.sh "${diar_opts[@]}" --stage "${stage}" \
        --stop_stage "$((stop_stage < 4 ? stop_stage : 4))" \
        --diar_args "--batch_size 1"
fi

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    local/train_windowed.sh --config "${train_config}" --diar_tag "${diar_tag}" \
        --train_set "${train_set}" --valid_set "${valid_set}" --ngpu "${ngpu}"
fi

if [ "${stop_stage}" -ge 6 ]; then
    ./diar.sh "${diar_opts[@]}" --stage "$((stage > 6 ? stage : 6))" \
        --stop_stage "${stop_stage}" \
        --diar_tag "${diar_tag}" --inference_model "${inference_model}"
fi

if [ "${stage}" -le 7 ] && [ "${stop_stage}" -ge 7 ]; then
    # every set at one threshold and median filter, chosen on AMI Mix-Headset dev
    python3 local/show_vad_result.py "exp/diar_${diar_tag}" --tune_set ami_mh_dev
fi
