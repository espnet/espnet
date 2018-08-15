#!/bin/bash

# Copyright 2018  Hirofumi Inaguma
#           2018  Kyoto Univerity (author: Hirofumi Inaguma)
# Apache 2.0

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <file> <lang>"
  exit 1
fi

set -e
file=$1
lang=$2

normalize-punctuation.perl -l $lang < $file | \
  tokenizer.perl -a -l $lang
