#!/usr/bin/env bash
# frontia_vad_v2: voice activity detection trained as single-output diarization
# (diar.sh --num_spk 1) on many domains, balanced per epoch by
# --chunk_upsampling_factor (conf/train.yaml) over the domains in utt2category.
#
#   ./run.sh                    data preparation (local/data.sh), then diar.sh
#   ./run.sh --stage 4          training and evaluation only
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
. utils/parse_options.sh

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
    --skip_dumped true
)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    local/data.sh
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # dumps the test sets; train and dev are marked done by local/data.sh
    ./diar.sh "${diar_opts[@]}" --stage 2 --stop_stage 2
fi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    ./diar.sh "${diar_opts[@]}" --stage 4 --stop_stage "${stop_stage}" "$@"
fi
