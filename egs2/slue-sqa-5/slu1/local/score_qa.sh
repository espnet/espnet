. ./db.sh
log_dir=$1 # directory to save post-processed output and results
token_fn=$2 # decoded frame-level outputs
split=$3
decoder=$4 # 1:use decoder output , 0: use ctc greedy output
tune=$5 # 1: tune, 0: don't tune
frame_len=$6
if [ $# -gt 6 ]; then
	decoder_token_fn=$7  # decoded outputs using joint CTC attention decoding
fi

# see README for more details on frame_len
# For example: frame_len=4e-2 for wav2vec2.0, wavLM, and HuBERT backbone with conv2d2 conformer prediction head

if ! test -f "data/${split}/all_word_alignments.json"; then
    python local/get_word_alignments.py ${SQA_5}
fi
if ! test -f "data/${split}/timestamp"; then
    python local/get_timestamps.py ${SQA_5}
fi

if [ "$decoder" -eq 1 ]; then
    python local/write_decoder_ctc_outputs.py --token_fn $decoder_token_fn --log_dir $log_dir --split $split --frame_len $frame_len --ctc_token_fn $token_fn
else
    python local/reformat_ctc_outputs.py --token_fn $token_fn --log_dir $log_dir --split $split --frame_len $frame_len
fi

if [ "$tune" -eq 1 ]; then
    for offset in $(seq -0.3 0.02 0.3); do
        echo "Evaluating offset ${offset}"
        python local/score_qa.py evaluate_submission --log_dir $log_dir --split devel --ms False --offset $offset
    done
    python local/score_qa.py choose_best --log_dir $log_dir
    if [ "$split" = "test" ]; then
        python local/score_qa.py eval_test --log_dir $log_dir
    fi
elif [ "$split" = "test" ]; then
    python local/score_qa.py eval_test --log_dir $log_dir
else
    offset=0.0
    python local/score_qa.py evaluate_submission --log_dir $log_dir --split $split --ms False --offset 0.0
fi
