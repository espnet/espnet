#!/usr/bin/env bash

. path.sh


# check if normalize-punctuation.perl exists
if ! command -v <normalize-punctuation.perl> &> /dev/null
then
    echo "normalize-punctuation.perl could not be found. Compile moses in espnet/tools as make moses.done"
    exit 1
fi

# remove punctuations from text
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
fi

# Example: ./local/text_normlize.sh <path_to_text_dir>
# Search for text files with _text.txt extension and normalize.
text_dir=$1

for root in ${text_dir}/*; do
    for f in ${root}/*_text.txt; do
        cut -d " " -f 1 ${f} > ${f}.uttid
        cut -d " " -f 2- ${f} > ${f}.trans

        normalize-punctuation.perl -l en < ${f}.trans > ${f}.trans.norm
        scripts/utils/remove_punctuation.pl < ${f}.trans.norm > ${f}.trans.norm.rm

        paste -d " " ${f}.uttid ${f}.trans.norm.rm > ${f}.new
        mv ${f} ${f}.old
        mv ${f}.new ${f}

        rm ${f}.uttid ${f}.trans ${f}.trans.norm ${f}.trans.norm.rm
        echo "Punctuation normalization completed: $f"
    done
done
