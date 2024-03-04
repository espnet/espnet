dir_name=$1
split="test"

# if ! test -f "data/nel_gt/${split}_all_word_alignments.json"; then
#     python local/prepare_nel_data.py
# fi
python local/reformat_ctc_outputs.py --dir_name $dir_name --split $split
# python local/write_decoder_ctc_outputs.py --dir_name $dir_name --split $split
# python local/write_decoder_ctc_outputs.py --dir_name $dir_name --split $split

offset=0.02
python local/score_qa.py evaluate_submission --dir_name $dir_name --split $split --ms False --offset 0.02

# uncomment the lines below to tune offset parameter

# for offset in $(seq -0.3 0.02 0.3); do
#     python local/score_nel.py evaluate_submission --dir_name $dir_name --split $split --ms False --offset $offset
# done

# python local/score_nel.py choose_best --dir_name $dir_name
