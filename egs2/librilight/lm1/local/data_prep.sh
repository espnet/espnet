#!/usr/bin/env bash


data_dir="data"
dump_dir="dump"

tokens=${data_dir}/tokens.txt
if ! [ -f "$tokens" ]; then
    echo "${tokens} file does not exist. Please prepare ${tokens} file."
    exit 1
fi
# Copy tokens.txt
mkdir -p ${data_dir}/token_list/word
cp ${tokens} data/token_list/word/

lm_train_text=${data_dir}/lm_train.txt
if ! [ -f "$lm_train_text" ]; then
    echo "${lm_train_text} file does not exist. Please prepare ${lm_train_text} file."
    exit 1
fi
# Prepare dump
data_feats=${dump_dir}/raw
mkdir -p ${data_feats}
cat ${lm_train_text} | awk ' { if( NF != 1 ) print $0; } ' > "${data_feats}/lm_train.txt"
