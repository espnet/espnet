#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. utils/parse_options.sh || exit 1;

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <src-dir> <dst-dir>"
    echo "e.g.: $0 /home/ws18hinag/libri_french/dev data/dev"
    exit 1
fi

src=$1
set=$(echo $2 | awk -F"/" '{print $NF}')
dst=$(pwd)/data/local/${set}

mkdir -p ${dst} || exit 1;

[ ! -d ${src} ] && echo "$0: no such directory ${src}" && exit 1;

wav_scp=${dst}/wav.scp; [[ -f "${wav_scp}" ]] && rm ${wav_scp}
trans=${dst}/text; [[ -f "${trans}" ]] && rm ${trans}
utt2spk=${dst}/utt2spk; [[ -f "${utt2spk}" ]] && rm ${utt2spk}

# error check
n_wav=$(find -L ${src}/audiofiles -iname "*.wav" | wc -l)
n_en=$(cat ${src}/${set}.en | wc -l)
n_fr1=$(cat ${src}/${set}.fr | wc -l)
n_fr2=$(cat ${src}/${set}_gtranslate.fr | wc -l)
[ ${n_wav} -ne ${n_en} ] && echo "Warning: expected ${n_wav} data data files, found ${n_en}" && exit 1;
[ ${n_wav} -ne ${n_fr1} ] && echo "Warning: expected ${n_wav} data data files, found ${n_fr1}" && exit 1;
[ ${n_wav} -ne ${n_fr2} ] && echo "Warning: expected ${n_wav} data data files, found ${n_fr2}" && exit 1;

# extract meta data
sed -e 's/\s\+/ /g' ${src}/alignments.meta | sed -e '1d' | while read line; do
    file_name=$(echo $line | cut -f 5 -d " ")
    if [ ${set} = other ] && [ $(soxi -t ${src}/audiofiles/${file_name}.wav) = flac ]; then
        # NOTE: some utterances in other directory are flac rather than wav files
        echo ${file_name} | awk -v "dir=${src}/audiofiles" '{printf "%s flac -c -d -s %s/%s.wav |\n", $0, dir, $0}' >> ${wav_scp} || exit 1;
    else
        echo ${file_name} | awk -v "dir=${src}/audiofiles" '{printf "%s cat %s/%s.wav |\n", $0, dir, $0}' >> ${wav_scp} || exit 1;
    fi

    book=$(basename ${file_name} | cut -f 1 -d "-")
    chapter=$(basename ${file_name} | cut -f 2 -d "-")
    echo ${file_name} | awk -v "book=$book" -v "chapter=$chapter" '{printf "%s %s-%s\n", $1, book, chapter}' >> ${utt2spk} || exit 1;
done

function clean-dash {
    perl -pe 's/^(-+)([^\s-])/$1 $2/g'
}
function clean-quotes {
    perl -pe "s/\"\s*\"/\"/g"
}

for lang in en fr fr.gtranslate; do
    if [ ${lang} = fr.gtranslate ]; then
        cp ${src}/${set}_gtranslate.fr ${dst}/${lang}.org
    else
        cp ${src}/${set}.${lang} ${dst}/${lang}.org
    fi

    # normalize punctuation
    if [ ${lang} = fr.gtranslate ]; then
        if [ ${set} = "train" ] || [ ${set} = "other" ]; then
            clean-quotes < ${dst}/${lang}.org | clean-dash | normalize-punctuation.perl -l fr | sed -e "s/•/·/g" | local/normalize_punctuation.pl > ${dst}/${lang}.norm
        else
            clean-dash < ${dst}/${lang}.org | normalize-punctuation.perl -l fr > ${dst}/${lang}.norm
        fi
    else
        if [ ${set} == "train" ] || [ ${set} = "other" ]; then
            clean-quotes < ${dst}/${lang}.org | clean-dash | normalize-punctuation.perl -l ${lang} | sed -e "s/•/·/g" | local/normalize_punctuation.pl > ${dst}/${lang}.norm
        else
            clean-dash < ${dst}/${lang}.org | normalize-punctuation.perl -l ${lang} > ${dst}/${lang}.norm
        fi
    fi
    # NOTE: only Moses script and clean-dash are applied for evaluation sets
    # See https://github.com/eske/seq2seq/blob/master/config/LibriSpeech/prepare-raw.sh

    # lowercasing
    lowercase.perl < ${dst}/${lang}.norm > ${dst}/${lang}.norm.lc
    cp ${dst}/${lang}.norm ${dst}/${lang}.norm.tc

    # remove punctuation (not used)
    local/remove_punctuation.pl < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.rm

    # tokenization
    if [ ${lang} = fr.gtranslate ]; then
        tokenizer.perl -l fr -q < ${dst}/${lang}.norm.tc > ${dst}/${lang}.norm.tc.tok
        tokenizer.perl -l fr -q < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.tok
        tokenizer.perl -l fr -q < ${dst}/${lang}.norm.lc.rm > ${dst}/${lang}.norm.lc.rm.tok
    else
        tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.tc > ${dst}/${lang}.norm.tc.tok
        tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.tok
        tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.lc.rm > ${dst}/${lang}.norm.lc.rm.tok
    fi
    paste -d " " <(awk '{print $1}' ${wav_scp}) <(cat ${dst}/${lang}.norm.tc.tok | awk '{if(NF>0) {print $0;} else {print "emptyuttrance";}}') \
        > ${dst}/text.tc.${lang}
    paste -d " " <(awk '{print $1}' ${wav_scp}) <(cat ${dst}/${lang}.norm.lc.tok | awk '{if(NF>0) {print $0;} else {print "emptyuttrance";}}') \
        > ${dst}/text.lc.${lang}
    paste -d " " <(awk '{print $1}' ${wav_scp}) <(cat ${dst}/${lang}.norm.lc.rm.tok | awk '{if(NF>0) {print $0;} else {print "emptyuttrance";}}') \
        > ${dst}/text.lc.rm.${lang}

    # save original and cleaned punctuation
    text2token.py -s 0 -n 1 ${dst}/${lang}.org | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.${lang}
    text2token.py -s 0 -n 1 ${dst}/${lang}.norm.tc | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.clean.${lang}
done


spk2utt=${dst}/spk2utt
utils/utt2spk_to_spk2utt.pl <${utt2spk} >$spk2utt || exit 1

utils/fix_data_dir.sh --utt_extra_files "text.tc.en text.tc.fr text.tc.fr.gtranslate \
                                         text.lc.en text.lc.fr text.lc.fr.gtranslate \
                                         text.lc.rm.en text.lc.rm.fr text.lc.rm.fr.gtranslate" ${dst} || exit 1;
utils/validate_data_dir.sh --no-feats --no-text ${dst} || exit 1;


# error check
for f in text.tc.en text.tc.fr text.tc.fr.gtranslate; do
    ntrans=$(wc -l <${dst}/${f})
    nutt2spk=$(wc -l <${utt2spk})
    ! [ "${ntrans}" -eq "${nutt2spk}" ] && \
        echo "Inconsistent #transcripts(${ntrans}) and #utt2spk(${nutt2spk})" && exit 1;
done


# Copy stuff intoc its final locations [this has been moved from the format_data script]
mkdir -p data/${set}
for f in spk2utt utt2spk wav.scp; do
    cp ${dst}/${f} data/${set}/${f}
done
# en
cp ${dst}/text.tc.en data/${set}/text.tc.en
cp ${dst}/text.lc.en data/${set}/text.lc.en
cp ${dst}/text.lc.rm.en data/${set}/text.lc.rm.en
# fr
cp ${dst}/text.tc.fr data/${set}/text.tc.fr
cp ${dst}/text.lc.fr data/${set}/text.lc.fr
cp ${dst}/text.lc.rm.fr data/${set}/text.lc.rm.fr
# fr.gtranslate
cp ${dst}/text.tc.fr.gtranslate data/${set}/text.tc.fr.gtranslate
cp ${dst}/text.lc.fr.gtranslate data/${set}/text.lc.fr.gtranslate
cp ${dst}/text.lc.rm.fr.gtranslate data/${set}/text.lc.rm.fr.gtranslate


# remove empty and sort utterances
cp -rf data/${set} data/${set}.tmp
grep -v emptyuttrance data/${set}/text.tc.en | cut -f 1 -d " " | sort > data/${set}/reclist.en
grep -v emptyuttrance data/${set}/text.tc.fr | cut -f 1 -d " " | sort > data/${set}/reclist.fr.org
grep -v emptyuttrance data/${set}/text.tc.fr.gtranslate | cut -f 1 -d " " | sort > data/${set}/reclist.fr.gtranslate
comm -12 data/${set}/reclist.fr.org data/${set}/reclist.fr.gtranslate > data/${set}/reclist.fr
comm -12 data/${set}/reclist.en data/${set}/reclist.fr > data/${set}/reclist
reduce_data_dir.sh data/${set}.tmp data/${set}/reclist data/${set}
utils/fix_data_dir.sh --utt_extra_files "text.tc.en text.tc.fr text.tc.fr.gtranslate \
                                         text.lc.en text.lc.fr text.lc.fr.gtranslate \
                                         text.lc.rm.en text.lc.rm.fr text.lc.rm.fr.gtranslate" data/${set}
rm -rf data/${set}.tmp


echo "$0: successfully prepared data in ${dst}"
exit 0;
