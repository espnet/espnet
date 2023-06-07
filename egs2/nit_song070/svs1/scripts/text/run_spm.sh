#!/usr/bin/env bash

# This script is called in data preparation step by local/data.sh
# It takes the data prepared using token type word as input
# It then trains a bpe model with "nbpe" number of tokens on the train transcript i.e. text after first word (intent)
# It then encodes the transcript for train, valid and test using the trained bpe model
nbpe=500 #try 100, 500, 1000
bpemode=bpe #try unigram, bpe

. utils/parse_options.sh

new_data=data_${bpemode}_${nbpe}
bpemodel=${new_data}/spm_train_${bpemode}${nbpe}

cp -R data ${new_data}

cut -d' ' -f2 data/train/text | sort | uniq > ${new_data}/intents.txt
cut -d' ' -f3- data/train/text > ${new_data}/input.txt

spm_train --input=${new_data}/input.txt \
            --model_prefix=${bpemodel} \
            --vocab_size=${nbpe} \
            --character_coverage=1.0 \
            --model_type=${bpemode} \
            --model_prefix=${bpemodel} \
            --input_sentence_size=100000000 \
            --bos_id=-1 \
            --eos_id=-1 \
            --unk_id=0

for split in train devel test; do
    cut -d' ' -f-2 data/${split}/text > ${new_data}/tmp_${split}_utt
    cut -d' ' -f3- data/${split}/text > ${new_data}/tmp_${split}_transcript
    spm_encode --model=${bpemodel}.model --output_format=piece < ${new_data}/tmp_${split}_transcript > ${new_data}/new_${split}_transcript
    paste -d' ' ${new_data}/tmp_${split}_utt ${new_data}/new_${split}_transcript > ${new_data}/${split}/text
    rm ${new_data}/tmp_${split}_utt
    rm ${new_data}/tmp_${split}_transcript
    rm ${new_data}/new_${split}_transcript
done
