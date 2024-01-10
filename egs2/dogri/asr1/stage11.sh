#!/bin/bash

# Command to run
command="./asr.sh --stage 11 --stop_stage 11 --train_set train --valid_set dev --test_sets test --pretrained_model "/home/rathna/espnet/egs2/e_branchformer_librispeech/exp/asr_train_asr_e_branchformer_raw_en_bpe5000_sp/valid.acc.ave_10best.pth" --use_ngram true --asr_config \"conf/train_asr_e_branchformer.yaml\" --ignore_init_mismatch "true" --use_lm true --ngpu 4 --nbpe 256"

# Output file
output_file="output.log"

# Run the command and redirect the output to the file
$command > $output_file 2>&1

# Check the exit status of the command
if [ $? -eq 0 ]; then
    echo "Command executed successfully. Output saved in $output_file"
else
    echo "Error: Command failed. Check $output_file for details."
fi
