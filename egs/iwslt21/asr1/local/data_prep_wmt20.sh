#!/bin/bash

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

max_length=80
length_ratio=1.5

. utils/parse_options.sh || exit 1;

datasize=$1

if [ ${datasize} = "5m" ]; then
    prefix=subset5000000
    download_url="https://www.cs.jhu.edu/~kevinduh/t/iwslt21/wmt20/wmt20-de-en-subset5m.tgz"
elif [ ${datasize} = "10m" ]; then
    prefix=subset10000000
    download_url="https://www.cs.jhu.edu/~kevinduh/t/iwslt21/wmt20/wmt20-de-en-subset10m.incl_paracrawl.tgz"
elif [ ${datasize} = "20m" ]; then
    prefix=subset20000000
    download_url="https://www.cs.jhu.edu/~kevinduh/t/iwslt21/wmt20/wmt20-de-en-subset20m.incl_paracrawl.tgz"
else
    echo "${datasize} is not supported."
    exit 1;
fi

dst=data/local/downloads/wmt20_subset${datasize}
mkdir -p ${dst}

mkdir -p data/local/downloads
if [ ! -f data/local/downloads/wmt20-de-en-subset${datasize}.tgz ]; then
    wget ${download_url} -O data/local/downloads/wmt20-de-en-subset${datasize}.tgz || exit 1;
    tar -xzvf data/local/downloads/wmt20-de-en-subset${datasize}.tgz -C ${dst} || exit 1;
fi

mkdir -p data/local/tools
if [ ! -d data/local/tools/langid.py ]; then
    git clone https://github.com/saffsd/langid.py data/local/tools/langid.py || exit 1;
fi

# error check
n_en=$(cat ${dst}/${prefix}.en | wc -l)
n_de=$(cat ${dst}/${prefix}.de | wc -l)
[ ${n_en} -ne ${n_de} ] && echo "Warning: expected ${n_en} data files, found ${n_de}" && exit 1;

for lang in en de; do
    cat ${dst}/${prefix}.${lang} | \
        remove-non-printing-char.perl > ${dst}/${lang}.norm
    awk '{printf "wmt20-%08d %s\n", NR, $0 }' ${dst}/${lang}.norm | sort > ${dst}/text.${lang} || exit 1;
done
cut -d " " -f 1 ${dst}/text.en | awk '{print $0" "$0}'> ${dst}/utt2spk
cut -d " " -f 1 ${dst}/text.en | awk '{print $0" "$0}'> ${dst}/spk2utt
cp ${dst}/text.en ${dst}/text  # dummy

# language identification w/ langid.py
for lang in en de; do
    paste -d " " <(cut -d" " -f 1 ${dst}/text.${lang}) <(cut -d" " -f 2- ${dst}/text.${lang} | python data/local/tools/langid.py/langid/langid.py --line -n) > ${dst}/langidpy.${lang}
    # NOTE: tokenization is included in langid.py
    grep " ('${lang}'," ${dst}/langidpy.${lang} | cut -d" " -f 1 > ${dst}/reclist.${lang}
done

# extract common lines
comm -12 ${dst}/reclist.en ${dst}/reclist.de > ${dst}/reclist
reduce_data_dir.sh ${dst} ${dst}/reclist ${dst}.langidpy || exit 1;
cp ${dst}/text.en ${dst}.langidpy
cp ${dst}/text.de ${dst}.langidpy
utils/fix_data_dir.sh --utt_extra_files "text.en text.de" ${dst}.langidpy || exit 1;
dst=${dst}.langidpy

for lang in en de; do
    # tokenization
    cut -d " " -f 2- ${dst}/text.${lang} | tokenizer.perl -threads 8 -l ${lang} -q > ${dst}/${lang}.tok

    paste -d " " <(cut -d " " -f 1 ${dst}/text.${lang}) <(cat ${dst}/${lang}.tok) > ${dst}/text.tok.${lang}
done

# length filtering
clean-corpus-n.perl -ratio ${length_ratio} ${dst}/text.tok en de ${dst}/text.tok.clean 1 ${max_length} || exit 1;

# character filtering
for lang in en de; do
    wc -l ${dst}/text.tok.clean.${lang}
    local/filter_parentheses.py ${dst}/text.tok.clean.${lang} > ${dst}/text.tc.${lang} || exit 1;
    wc -l ${dst}/text.tc.${lang}
done

# lowercasing, remove punctuation
for lang in en de; do
    paste -d " " <(cut -d " " -f 1 ${dst}/text.tc.${lang}) <(cut -d " " -f 2- ${dst}/text.tc.${lang} | lowercase.perl) > ${dst}/text.lc.${lang}
    paste -d " " <(cut -d " " -f 1 ${dst}/text.lc.${lang}) <(cut -d " " -f 2- ${dst}/text.lc.${lang} | remove_punctuation.pl) > ${dst}/text.lc.rm.${lang}
    cut -d" " -f 2- ${dst}/text.tc.${lang} | text2token.py -s 0 -n 1 | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.clean.${lang}
done

# extract common lines again
for lang in en de; do
    cut -d" " -f 1 ${dst}/text.tc.${lang} > ${dst}/reclist.${lang}
done
comm -12 ${dst}/reclist.en ${dst}/reclist.de > ${dst}/reclist
reduce_data_dir.sh ${dst} ${dst}/reclist data/tr_wmt20_subset${datasize} || exit 1;
for lang in en de; do
    for case in lc.rm lc tc; do
        cp ${dst}/text.${case}.${lang} data/tr_wmt20_subset${datasize}
    done
done
utils/fix_data_dir.sh --utt_extra_files "text.tc.en text.lc.en text.lc.rm.en \
                                         text.tc.de text.lc.de text.lc.rm.de" data/tr_wmt20_subset${datasize} || exit 1;
