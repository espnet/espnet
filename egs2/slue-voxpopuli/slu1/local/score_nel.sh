log_dir=$1 # directory to save post-processed output and results
token_fn=$2 # decoded frame-level outputs
split=$3 # devel or test
tune=$4 # 1: tune, 0: don't tune
frame_len=$5

# see README for more details on frame_len
# For example: frame_len=4e-2 for wav2vec2.0, wavLM, and HuBERT backbone with conv2d2 conformer prediction head

if ! test -f "data/nel_gt/${split}_all_word_alignments.json"; then
    python local/prepare_nel_data.py
fi

python local/reformat_ctc_outputs.py --token_fn $token_fn --log_dir $log_dir --split $split --frame_len $frame_len

if [ "$tune" -eq 1 ]; then
    for offset in $(seq -0.3 0.02 0.3); do
        echo "Evaluating offset ${offset}"
        python local/score_nel.py evaluate_submission --log_dir $log_dir --split devel --ms False --offset $offset
    done
    python local/score_nel.py choose_best --log_dir $log_dir
    if [ "$split" = "test" ]; then
        python local/score_nel.py eval_test --log_dir $log_dir
    fi
elif [ "$split" = "test" ]; then
    python local/score_nel.py eval_test --log_dir $log_dir
else
    offset=0.0
    python local/score_nel.py evaluate_submission --log_dir $log_dir --split $split --ms False --offset 0.0
fi
