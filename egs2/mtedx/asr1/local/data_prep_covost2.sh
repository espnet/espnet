#!/usr/bin/env bash

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. ./path.sh || exit 1;

. utils/parse_options.sh || exit 1;

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <translation-dir> <src-lang> <tgt-lang>"
    echo "e.g.: $0 downloads/translation source_lang target_lang"
    exit 1;
fi

covost2_datadir=$1
src_lang=$2
tgt_lang=$3


tsv_path=${covost2_datadir}/covost_v2.${src_lang}_${tgt_lang}.tsv
[ ! -f ${tsv_path} ] && echo "$0: no such directory ${tsv_path}" && exit 1;
data_dir=data/validated.${src_lang}

for set in train dev test; do
    dst=data/local/${set}.${src_lang}-${tgt_lang}
    mkdir -p ${dst} || exit 1;

    src=${dst}/text.${src_lang}
    tgt=${dst}/text.${tgt_lang}

    # extract translation from CoVoST2 tsv file and align to transcription in CommonVoice
    python3 local/process_tsv.py ${tsv_path} ${data_dir}/text ${data_dir}/utt2spk ${src} ${tgt} ${set} || exit 1;
    utils/utt2spk_to_spk2utt.pl < ${data_dir}/utt2spk > ${dst}/spk2utt
    utils/spk2utt_to_utt2spk.pl < ${dst}/spk2utt > ${dst}/utt2spk
    cp ${data_dir}/wav.scp ${dst}/wav.scp

    # sort
    sort ${src} > ${src}.tmp
    sort ${tgt} > ${tgt}.tmp
    mv ${src}.tmp ${src}
    mv ${tgt}.tmp ${tgt}

    # error check
    n_src=$(cat ${src} | wc -l)
    n_tgt=$(cat ${tgt} | wc -l)
    [ ${n_src} -ne ${n_tgt} ] && echo "Warning: expected ${n_tgt} data files, found ${n_src} in ${src_lang}-${tgt_lang}"
    #  && exit 1;

    cut -f 2- -d " " ${src} > ${dst}/${src_lang}.org
    cut -f 2- -d " " ${tgt} > ${dst}/${tgt_lang}.org
    cut -f 1 -d " " ${src} > ${dst}/reclist.${src_lang}
    cut -f 1 -d " " ${tgt} > ${dst}/reclist.${tgt_lang}

    for lang in ${src_lang} ${tgt_lang}; do
        lang_trim="$(echo "${lang}" | cut -f 1 -d '-')"

        # normalize punctuation
        if [ ${lang} = ${src_lang} ]; then
            lowercase.perl < ${dst}/${lang}.org > ${dst}/${lang}.org.lc
            # NOTE: almost all characters in transcription on CommonVoice is truecased
            normalize-punctuation.perl -l ${lang_trim} < ${dst}/${lang}.org.lc > ${dst}/${lang}.norm
        else
            normalize-punctuation.perl -l ${lang_trim} < ${dst}/${lang}.org > ${dst}/${lang}.norm
        fi

        # lowercasing
        lowercase.perl < ${dst}/${lang}.norm > ${dst}/${lang}.norm.lc
        cp ${dst}/${lang}.norm ${dst}/${lang}.norm.tc

        # remove punctuation
        remove_punctuation.pl < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.rm

        for case in lc.rm lc tc; do
            # tokenization
            tokenizer.perl -l ${lang_trim} -q < ${dst}/${lang}.norm.${case} > ${dst}/${lang}.norm.${case}.tok

            paste -d " " ${dst}/reclist.${lang} ${dst}/${lang}.norm.${case}.tok | sort > ${dst}/text.${case}.${lang}
        done

        # save original and cleaned punctuation
        lowercase.perl < ${dst}/${lang}.org | text2token.py -s 0 -n 1 | tr " " "\n" \
            | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.${lang}
        lowercase.perl < ${dst}/${lang}.norm.tc | text2token.py -s 0 -n 1 | tr " " "\n" \
            | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.clean.${lang}
    done

    # extract common lines
    comm -12 <(sort ${dst}/reclist.${src_lang}) <(sort ${dst}/reclist.${tgt_lang}) > ${dst}/reclist

    # Copy stuff intoc its final locations [this has been moved from the format_data script]
    reduce_data_dir.sh ${dst} ${dst}/reclist data/${set}.${src_lang}-${tgt_lang}
    for case in lc.rm lc tc; do
        cp ${dst}/text.${case}.${src_lang} data/${set}.${src_lang}-${tgt_lang}
        cp ${dst}/text.${case}.${tgt_lang} data/${set}.${src_lang}-${tgt_lang}
    done
    utils/fix_data_dir.sh --utt_extra_files \
        "text.tc.${src_lang} text.lc.${src_lang} text.lc.rm.${src_lang} \
         text.tc.${tgt_lang} text.lc.${tgt_lang} text.lc.rm.${tgt_lang}" data/${set}.${src_lang}-${tgt_lang}

    echo "$0: successfully prepared data in ${dst}"
done
