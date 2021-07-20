#!/bin/bash

mkdir -p data/lang data/local/dict

cat corpus/lang/lexicon.txt | sed '1,4d' > data/local/dict/lexicon_words.txt

cp corpus/lang/* data/local/dict/.

touch data/local/dict/extra_questions.txt

echo "<unk>" > data/lang/oov.txt

echo "Dictionary preparation succeeded"
