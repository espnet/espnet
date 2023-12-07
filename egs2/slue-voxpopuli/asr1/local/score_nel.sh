dir_name=$1
split="devel"

if ! test -f "data/nel_gt/${split}_all_word_alignments.json"; then
    python local/prepare_nel_data.py 
fi
python local/reformat_ctc_outputs.py --dir_name $dir_name --split $split

# offset=0.0
python local/score_nel.py evaluate_submission --dir_name $dir_name --split $split --ms False --offset 0.0

# uncomment the lines below to tune offset parameter

# for offset in $(seq -0.3 0.02 0.3); do
#     python codes/eval.py evaluate_submission --dir_name $dir_name --split $split --ms False --offset $offset
# done

# python codes/eval.py choose_best --dir_name $dir_name