#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

asr_speech_fold_length=1000 # 6.25 sec, because audio is 5 sec each.
inference_model=valid.acc.best.pth

n_folds=5 # This runs all 5 folds in parallel, take care.

asr_config=conf/beats_classification.yaml

mynametag=fast.fold

# # NOTE(shikhar): Abusing variable lang to store fold number.
for fold in $(seq 1 $n_folds); do
    train_set="train${fold}"
    valid_set="val${fold}"
    test_set="val${fold}"
    ./asr.sh \
        --local_data_opts "${fold}" \
        --asr_tag "${mynametag}${fold}" \
        --lang "${fold}" \
        --ngpu 1 \
        --stage 15 \
        --inference_args "--ctc_weight 0.0 --maxlenratio -1" \
        --token_type word \
        --asr_speech_fold_length ${asr_speech_fold_length} \
        --use_lm false \
        --feats_type raw \
        --max_wav_duration 6 \
        --feats_normalize utterance_mvn\
        --inference_nj 8 \
        --inference_asr_model ${inference_model} \
        --asr_config "${asr_config}" \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_set}" "$@" &
done

wait

# Average the accuracies of all folds.
# Please ensure that the RESULTS.md file is empty before running this script.
total_sum=0
total_count=0
for i in $(seq 1 $n_folds); do
    values=$(grep "val${i}" exp/asr_${mynametag}${i}/RESULTS.md | head -n 1 | awk -F'|' '{print $(NF-1)}')
    for value in $values; do
        total_sum=$(echo "$total_sum + $value" | bc)
        total_count=$((total_count + 1))
        break
    done
done

if [ $total_count -gt 0 ]; then
    average=$(echo "scale=2; $total_sum / $total_count" | bc)
    echo "Avg. acc: $(echo "100 - $average" | bc)"
    echo "Over $total_count folds."
else
    echo "No values found."
fi
