#!/usr/bin/env bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

copy_fbank=true

. utils/parse_options.sh || exit 1;

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <src-dir>"
    echo "e.g.: $0 /export/a13/kduh/mtdata/how2/how2-300h-v1"
    exit 1;
fi

# copy fbank features
if [ ${copy_fbank} = true ]; then
    echo "copy fbank features..."
    fbank=$1/features/fbank_pitch_181506
    mkdir -p fbank
    cp -rf ${fbank}/* fbank

    # replace ark path
    for file in $(\find fbank -maxdepth 1 -type f | grep scp); do
        fbank_path=$(pwd)"/fbank"
        fbank_path=$(echo ${fbank_path} | sed -e "s/\//@/g")
        sed -e "s/ARK_PATH/${fbank_path}/" < ${file} | sed -e "s/@/\//g" > ${file}.tmp
        mv ${file}.tmp ${file}
    done

    echo "successfully copied fbank features"
fi


for set in train val dev5; do
    src=$1/data/${set}
    dst=data/local/${set}

    [ ! -d ${src} ] && echo "$0: no such directory ${src}" && exit 1;

    mkdir -p ${dst} || exit 1;

    trans_en=${dst}/text.id.en; [[ -f "${trans_en}" ]] && rm ${trans_en}
    trans_pt=${dst}/text.id.pt; [[ -f "${trans_pt}" ]] && rm ${trans_pt}
    utt2spk=${dst}/utt2spk; [[ -f "${utt2spk}" ]] && rm ${utt2spk}
    spk2utt=${dst}/spk2utt; [[ -f "${spk2utt}" ]] && rm ${spk2utt}
    segments=${dst}/segments; [[ -f "${segments}" ]] && rm ${segments}
    feat=${dst}/feats.scp; [[ -f "${feat}" ]] && rm ${feat}

    # copy data directory
    [[ -d "${dst}" ]] && rm -rf ${dst}
    cp -rf ${src} ${dst}

    # change file name
    mv ${trans_en} ${dst}/text.en; trans_en=${dst}/text.en;
    mv ${trans_pt} ${dst}/text.pt; trans_pt=${dst}/text.pt;

    # error check
    n=$(wc -l < ${segments})
    n_en=$(cat ${trans_en} | wc -l)
    n_pt=$(cat ${trans_pt} | wc -l)
    [ ${n} -ne ${n_en} ] && echo "Warning: expected ${n} data data files, found ${n_en}" && exit 1;
    [ ${n} -ne ${n_pt} ] && echo "Warning: expected ${n} data data files, found ${n_pt}" && exit 1;

    cut -f 2- -d " " ${trans_en} > ${dst}/en.org
    cut -f 2- -d " " ${trans_pt} > ${dst}/pt.org

    for lang in en pt; do
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

        paste -d " " ${dst}/id ${dst}/${lang}.norm.tc.tok | sort > ${dst}/text.tc.${lang}
        paste -d " " ${dst}/id ${dst}/${lang}.norm.lc.tok | sort > ${dst}/text.lc.${lang}
        paste -d " " ${dst}/id ${dst}/${lang}.norm.lc.rm.tok | sort > ${dst}/text.lc.rm.${lang}

        # save original and cleaned punctuation
        lowercase.perl < ${dst}/${lang}.org | text2token.py -s 0 -n 1 | tr " " "\n" \
            | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.${lang}
        lowercase.perl < ${dst}/${lang}.norm.tc | text2token.py -s 0 -n 1 | tr " " "\n" \
            | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.clean.${lang}
    done


    # error check
    n=$(cat ${dst}/segments | wc -l)
    n_en=$(cat ${dst}/en.norm.tc.tok | wc -l)
    n_pt=$(cat ${dst}/pt.norm.tc.tok | wc -l)
    [ ${n} -ne ${n_en} ] && echo "Warning: expected ${n} data data files, found ${n_en}" && exit 1;
    [ ${n} -ne ${n_pt} ] && echo "Warning: expected ${n} data data files, found ${n_pt}" && exit 1;

    # replace ark path
    if [ ${copy_fbank} = true ]; then
        fbank_path=$(pwd)"/fbank"
        fbank_path=$(echo ${fbank_path} | sed -e "s/\//@/g")
        sed -e "s/ARK_PATH/${fbank_path}/" < ${feat} | sed -e "s/@/\//g" > ${feat}.tmp
        mv ${feat}.tmp ${feat}
    fi

    # Copy stuff into its final locations [this has been moved from the format_data script]
    mkdir -p data/${set}
    for f in spk2utt utt2spk; do
        cp ${dst}/${f} data/${set}/${f}
    done
    if [ ${copy_fbank} = true ]; then
        cp ${dst}/feats.scp data/${set}/feats.scp
    fi

    # NOTE: do not copy segments to pass utils/validate_data_dir.sh
    # en
    cp ${dst}/text.tc.en data/${set}/text.tc.en
    cp ${dst}/text.lc.en data/${set}/text.lc.en
    cp ${dst}/text.lc.rm.en data/${set}/text.lc.rm.en
    # pt
    cp ${dst}/text.tc.pt data/${set}/text.tc.pt
    cp ${dst}/text.lc.pt data/${set}/text.lc.pt
    cp ${dst}/text.lc.rm.pt data/${set}/text.lc.rm.pt

    echo "$0: successfully prepared data in ${dst}"
done
