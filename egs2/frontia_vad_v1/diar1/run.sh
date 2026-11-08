#!/usr/bin/env bash
# Voice activity detection trained as single-output diarization (diar.sh --num_spk 1),
# on 10 s chunks of the dumped recordings, each read on its own from the Omni-IO
# archive (conf/train.yaml: chunk_random_access), and evaluated on whole recordings. The default is the causal model (conf/train.yaml). The dual-mode
# model (online and offline with one set of weights):
#   ./run.sh --diar_config conf/tuning/train_dualmode.yaml --diar_tag train_dualmode
# scores it online (look-ahead 0). diar.sh writes inference into exp/diar_<diar_tag>,
# so the offline scores (look-ahead 0.99 s) go to a second directory with the same model:
#   mkdir -p exp/diar_train_dualmode_offline
#   ln -s ../diar_train_dualmode/{config.yaml,valid.acc.best.pth} exp/diar_train_dualmode_offline/
#   ./run.sh --stage 6 --diar_tag train_dualmode_offline \
#       --inference_config conf/tuning/decode_offline.yaml
# After stage 7, local/show_vad_result.py exp/diar_<diar_tag> reports every set at
# the threshold and median filter tuned on AMI Mix-Headset dev.
set -e
set -u
set -o pipefail

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

./diar.sh \
    --fs 16k \
    --audio_format omniio \
    --num_spk 1 \
    --collar 0.0 \
    --frame_shift 128 \
    --score_whole_recording true \
    --scoring_thresholds "0.3 0.4 0.5 0.6 0.7 0.8 0.9" \
    --stats_args "--batch_size 1" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --diar_config conf/train.yaml \
    --diar_tag train \
    --inference_config conf/decode.yaml \
    --local_data_opts "--corpora ${corpora}" \
    --skip_dumped true \
    "$@"
