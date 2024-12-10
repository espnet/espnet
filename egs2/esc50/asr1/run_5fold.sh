#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

asr_config=conf/beats_classification.yaml
asr_speech_fold_length=1000 # 6.25 sec, because audio is 5 sec each.

# inference_model=valid.acc.ave_5best.pth
inference_model=latest.pth

n_folds=5

# NOTE(shikhar): Abusing variable lang to store fold number.
for fold in $(seq 1 $n_folds); do
    train_set="train${fold}"
    valid_set="val${fold}"
    test_set="val${fold}"
    ./asr.sh \
        --asr_tag "fold${fold}" \
        --local_data_opts "${fold}" \
        --lang "${fold}" \
        --ngpu 1 \
        --stage 1 \
        --inference_args "--ctc_weight 0.0 --maxlenratio -1" \
        --token_type word \
        --asr_speech_fold_length ${asr_speech_fold_length} \
        --use_lm false \
        --feats_type raw \
        --max_wav_duration 6 \
        --feats_normalize utterance_mvn\
        --inference_nj 8 \
        --inference_asr_model latest.pth\
        --asr_config "${asr_config}" \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_set}" "$@" &
done

wait

# Average the accuracies of all folds.
total_sum=0
total_count=0
for i in $(seq 1 $n_folds); do
    values=$(grep "val${i}" exp/asr_fold${i}/RESULTS.md | head -n 1 | awk -F'|' '{print $(NF-1)}')
    for value in $values; do
        total_sum=$(echo "$total_sum + $value" | bc)
        total_count=$((total_count + 1))
        break
    done
done

if [ $total_count -gt 0 ]; then
    average=$(echo "scale=2; $total_sum / $total_count" | bc)
    echo "Avg. acc: $(echo "100 - $average" | bc)"
else
    echo "No values found."
fi
