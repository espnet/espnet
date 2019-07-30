#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. utils/parse_options.sh || exit 1;

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <src-dir>"
    echo "e.g.: $0 /export/corpora4/IWSLT/iwslt-corpus"
    exit 1;
fi

src=$1/train/iwslt-corpus
dst=data/local/train

[ ! -d ${src} ] && echo "$0: no such directory ${src}" && exit 1;

wav_dir=${src}/wav
trans_dir=${src}/parallel
yml=${trans_dir}/train.yaml
en=${trans_dir}/train.en
de=${trans_dir}/train.de

mkdir -p ${dst} || exit 1;

[ ! -d ${wav_dir} ] && echo "$0: no such directory ${wav_dir}" && exit 1;
[ ! -d ${trans_dir} ] && echo "$0: no such directory ${trans_dir}" && exit 1;
[ ! -f ${yml} ] && echo "$0: expected file ${yml} to exist" && exit 1;
[ ! -f ${en} ] && echo "$0: expected file ${en} to exist" && exit 1;
[ ! -f ${de} ] && echo "$0: expected file ${de} to exist" && exit 1;

wav_scp=${dst}/wav.scp; [[ -f "${wav_scp}" ]] && rm ${wav_scp}
trans_en=${dst}/text.en; [[ -f "${trans_en}" ]] && rm ${trans_en}
trans_de=${dst}/text.de; [[ -f "${trans_de}" ]] && rm ${trans_de}
utt2spk=${dst}/utt2spk; [[ -f "${utt2spk}" ]] && rm ${utt2spk}
spk2utt=${dst}/spk2utt; [[ -f "${spk2utt}" ]] && rm ${spk2utt}
segments=${dst}/segments; [[ -f "${segments}" ]] && rm ${segments}

# error check
n=$(cat ${yml} | grep duration | wc -l)
n_en=$(cat ${en} | wc -l)
n_de=$(cat ${de} | wc -l)
[ ${n} -ne ${n_en} ] && echo "Warning: expected ${n} data data files, found ${n_en}" && exit 1;
[ ${n} -ne ${n_de} ] && echo "Warning: expected ${n} data data files, found ${n_de}" && exit 1;


# (1a) Transcriptions and translations preparation
# make basic transcription file (add segments info)
cp ${yml} ${dst}/.yaml0
grep duration ${dst}/.yaml0 > ${dst}/.yaml1
awk '{
    duration=$3; offset=$5; spkid=$7;
    gsub(",","",duration);
    gsub(",","",offset);
    gsub(",","",spkid);
    gsub("spk.","",spkid);
    duration=sprintf("%.7f", duration);
    if ( duration < 0.2 ) extendt=sprintf("%.7f", (0.2-duration)/2);
    else extendt=0;
    offset=sprintf("%.7f", offset);
    startt=offset-extendt;
    endt=offset+duration+extendt;
    printf("ted_%04d_%07.0f_%07.0f\n", spkid, int(1000*startt+0.5), int(1000*endt+0.5));
}' ${dst}/.yaml1 > ${dst}/.yaml2
# NOTE: Extend the lengths of short utterances (< 0.2s) rather than exclude them

cp ${en} ${dst}/en.org
cp ${de} ${dst}/de.org

for lang in en de; do
    # normalize punctuation
    normalize-punctuation.perl -l ${lang} < ${dst}/${lang}.org | sed -e "s/‒/ - /g" | sed -e "s/‟/\"/g" | sed -e "s/，/,/g" | \
        local/normalize_punctuation.pl > ${dst}/${lang}.norm

    # lowercasing
    lowercase.perl < ${dst}/${lang}.norm > ${dst}/${lang}.norm.lc
    cp ${dst}/${lang}.norm ${dst}/${lang}.norm.tc

    # remove punctuation
    local/remove_punctuation.pl < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.rm

    # tokenization
    tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.tc > ${dst}/${lang}.norm.tc.tok
    tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.tok
    tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.lc.rm > ${dst}/${lang}.norm.lc.rm.tok

    if [ ${lang} = en ]; then
        paste -d " " ${dst}/.yaml2 ${dst}/${lang}.norm.tc.tok | sort > ${dst}/text.tc.${lang}
        paste -d " " ${dst}/.yaml2 ${dst}/${lang}.norm.lc.tok | sort > ${dst}/text.lc.${lang}
        paste -d " " ${dst}/.yaml2 ${dst}/${lang}.norm.lc.rm.tok | sort > ${dst}/text.lc.rm.${lang}
    else
        paste -d " " ${dst}/.yaml2 ${dst}/${lang}.norm.tc.tok | awk '{if (length($0) > 25) print $0}' | sort > ${dst}/text.tc.${lang}
        paste -d " " ${dst}/.yaml2 ${dst}/${lang}.norm.lc.tok | awk '{if (length($0) > 25) print $0}' | sort > ${dst}/text.lc.${lang}
        paste -d " " ${dst}/.yaml2 ${dst}/${lang}.norm.lc.rm.tok | awk '{if (length($0) > 25) print $0}' | sort > ${dst}/text.lc.rm.${lang}
        # NOTE: empty utterances are included in original German translations
    fi

    # save original and cleaned punctuation
    lowercase.perl < ${dst}/${lang}.org | text2token.py -s 0 -n 1 | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.${lang}
    lowercase.perl < ${dst}/${lang}.norm.tc | text2token.py -s 0 -n 1 | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.clean.${lang}
done


# error check
n=$(cat ${dst}/.yaml2 | wc -l)
n_en=$(cat ${dst}/en.norm.tc.tok | wc -l)
n_de=$(cat ${dst}/de.norm.tc.tok | wc -l)
[ ${n} -ne ${n_en} ] && echo "Warning: expected ${n} data data files, found ${n_en}" && exit 1;
[ ${n} -ne ${n_de} ] && echo "Warning: expected ${n} data data files, found ${n_de}" && exit 1;


# (1c) Make segments files from transcript
#segments file format is: utt-id start-time end-time, e.g.:
#ted_0001_0003501_0003684 ted_0001 003.501 0003.684
awk '{
    segment=$1; split(segment,S,"[_]");
    spkid=S[1] "_" S[2]; startf=S[3]; endf=S[4];
    printf("%s %s %.2f %.2f\n", segment, spkid, startf/1000, endf/1000);
}' < ${dst}/text.tc.de | sort > ${dst}/segments

awk '{
    segment=$1; split(segment,S,"[_]");
    spkid=S[1] "_" S[2];
    printf("%s cat '${wav_dir}'/%s_%d.wav |\n", spkid, S[1], S[2]);
}' < ${dst}/text.tc.de | uniq | sort > ${dst}/wav.scp

awk '{
    segment=$1; split(segment,S,"[_]");
    spkid=S[1] "_" S[2]; print $1 " " spkid
}' ${dst}/segments | sort > ${dst}/utt2spk

sort ${dst}/utt2spk | utils/utt2spk_to_spk2utt.pl | sort > ${dst}/spk2utt


# Match the number of utterances between En and De (reduce En utterances)
oldnum=$(wc -l ${dst}/text.tc.en | awk '{print $1}')
utils/filter_scp.pl ${dst}/utt2spk < ${dst}/text.tc.en > ${dst}/text.tc.en.tmp
newnum=$(wc -l ${dst}/text.tc.en.tmp | awk '{print $1}')
echo "change from ${oldnum} to ${newnum}"
rm ${dst}/text.tc.en
mv ${dst}/text.tc.en.tmp ${dst}/text.tc.en

# error check
n_en=$(cat ${dst}/text.tc.en | wc -l)
n_de=$(cat ${dst}/text.tc.de | wc -l)
[ ${n_en} -ne ${n_de} ] && echo "Warning: expected ${n_en} data data files, found ${n_de}" && exit 1;


# Copy stuff intoc its final locations [this has been moved from the format_data script]
mkdir -p data/train
for f in spk2utt utt2spk wav.scp segments; do
    cp ${dst}/${f} data/train/${f}
done
# en
cp ${dst}/text.tc.en data/train/text.tc.en
cp ${dst}/text.lc.en data/train/text.lc.en
cp ${dst}/text.lc.rm.en data/train/text.lc.rm.en
# de
cp ${dst}/text.tc.de data/train/text.tc.de
cp ${dst}/text.lc.de data/train/text.lc.de
cp ${dst}/text.lc.rm.de data/train/text.lc.rm.de


echo "$0: successfully prepared data in ${dst}"
