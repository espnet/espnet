#!/usr/bin/env bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. utils/parse_options.sh || exit 1;

download=data/local/download

# download test data
url=https://islpc21.is.cs.cmu.edu/ramons/iwslt2019.tar.gz
if [ -f ${download}/iwslt2019.tar.gz ]; then
    echo "${download}/iwslt2019.tar.gz exists and appears to be complete."
else
    if ! which wget >/dev/null; then
        echo "$0: wget is not installed."
        exit 1;
    fi
    echo "$0: downloading data from ${url}.  This may take some time, please be patient."

    if ! wget --no-check-certificate -P ${download} ${url}; then
        echo "$0: error executing wget ${url}"
        exit 1;
    fi
fi

if ! tar -zxvf ${download}/iwslt2019.tar.gz -C ${download}; then
    echo "$0: error un-tarring archive ${download}/iwslt2019.tar.gz"
    exit 1;
fi

dst=data/local/test_set_iwslt2019

mkdir -p ${dst} || exit 1;

trans_en=${dst}/text; [[ -f "${trans_en}" ]] && rm ${trans_en}
utt2spk=${dst}/utt2spk; [[ -f "${utt2spk}" ]] && rm ${utt2spk}
spk2utt=${dst}/spk2utt; [[ -f "${spk2utt}" ]] && rm ${spk2utt}
segments=${dst}/segments; [[ -f "${segments}" ]] && rm ${segments}
feat=${dst}/feats.scp; [[ -f "${feat}" ]] && rm ${feat}

# copy data directory
[[ -d "${dst}" ]] && rm -rf ${dst}
cp -rf ${download}/test_set_iwslt2019/test ${dst}

# error check
n=$(wc -l < ${segments})
n_en=$(cat ${trans_en} | wc -l)
[ ${n} -ne ${n_en} ] && echo "Warning: expected ${n} data data files, found ${n_en}" && exit 1;

cut -f 2- -d " " ${trans_en} > ${dst}/en.org

for lang in en; do
    # normalize punctuation
    normalize-punctuation.perl -l ${lang} < ${dst}/${lang}.org > ${dst}/${lang}.norm

    # lowercasing
    lowercase.perl < ${dst}/${lang}.norm > ${dst}/${lang}.norm.lc
    cp ${dst}/${lang}.norm ${dst}/${lang}.norm.tc

    # remove punctuation
    remove_punctuation.pl < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.rm

    # tokenization
    tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.tc > ${dst}/${lang}.norm.tc.tok
    tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.tok
    tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.lc.rm > ${dst}/${lang}.norm.lc.rm.tok

    paste -d " " <(cut -f 1 -d " " ${dst}/segments) <(cat ${dst}/${lang}.norm.tc.tok) | sort > ${dst}/text.tc.${lang}
    paste -d " " <(cut -f 1 -d " " ${dst}/segments) <(cat ${dst}/${lang}.norm.lc.tok) | sort > ${dst}/text.lc.${lang}
    paste -d " " <(cut -f 1 -d " " ${dst}/segments) <(cat ${dst}/${lang}.norm.lc.rm.tok) | sort > ${dst}/text.lc.rm.${lang}

    # save original and cleaned punctuation
    lowercase.perl < ${dst}/${lang}.org | text2token.py -s 0 -n 1 | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.${lang}
    lowercase.perl < ${dst}/${lang}.norm.tc | text2token.py -s 0 -n 1 | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.clean.${lang}
done


# error check
n=$(cat ${dst}/segments | wc -l)
n_en=$(cat ${dst}/en.norm.tc.tok | wc -l)
[ ${n} -ne ${n_en} ] && echo "Warning: expected ${n} data data files, found ${n_en}" && exit 1;

# replace ark path
fbank_path=$(pwd)"/fbank"
fbank_path=$(echo ${fbank_path} | sed -e "s/\//@/g")
sed -e "s/..\/fbank_pitch/${fbank_path}/" < ${feat} | sed -e "s/@/\//g" > ${feat}.tmp
mv ${feat}.tmp ${feat}

# Copy stuff into its final locations [this has been moved from the format_data script]
mkdir -p data/test_set_iwslt2019
for f in spk2utt utt2spk feats.scp; do
    cp ${dst}/${f} data/test_set_iwslt2019/${f}
done
# NOTE: do not copy segments to pass utils/validate_data_dir.sh
# en
cp ${dst}/text.tc.en data/test_set_iwslt2019/text.tc.en
cp ${dst}/text.lc.en data/test_set_iwslt2019/text.lc.en
cp ${dst}/text.lc.rm.en data/test_set_iwslt2019/text.lc.rm.en

echo "$0: successfully prepared data in ${dst}"

dst=data/local/test_set_iwslt2019


# copy fbank features
echo "copy fbank features..."
fbank=$download/test_set_iwslt2019/fbank_pitch
mkdir -p fbank
cp -rf ${fbank}/* fbank

# replace ark path
for file in $(\find fbank -maxdepth 1 -type f | grep scp); do
    fbank_path=`pwd`"/fbank"
    fbank_path=$(echo ${fbank_path} | sed -e "s/\//@/g")
    sed -e "s/.\//${fbank_path}\//" < ${file} | sed -e "s/@/\//g" > ${file}.tmp
    mv ${file}.tmp ${file}
done


echo "successfully copied fbank features"
