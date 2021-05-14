#!/usr/bin/env bash

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. utils/parse_options.sh || exit 1;

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <set>"
    echo "e.g.: $0 dev"
    exit 1
fi

set=$1
src=$(pwd)/data/local/mboshi-french-parallel-corpus/full_corpus_newsplit/${set}
dst=$(pwd)/data/local/${set}

[[ -d "${dst}" ]] && rm -rf ${dst}
mkdir -p ${dst} || exit 1;

# download data preparation scripts for transcriptions
[ ! -d data/local/mboshi-french-parallel-corpus ] && git clone https://github.com/besacier/mboshi-french-parallel-corpus data/local/mboshi-french-parallel-corpus

[ ! -d ${src} ] && echo "$0: no such directory ${src}" && exit 1;

wav_scp=${dst}/wav.scp; [[ -f "${wav_scp}" ]] && rm ${wav_scp}
trans=${dst}/text.*; [[ -f "${trans}" ]] && rm ${trans}
utt2spk=${dst}/utt2spk; [[ -f "${utt2spk}" ]] && rm ${utt2spk}

# error check
n_wav=$(find -L ${src} -iname "*.wav" | wc -l)
n_mb=$(find -L ${src} -iname "*.mb.cleaned" | wc -l)
n_fr=$(find -L ${src} -iname "*.fr.cleaned" | wc -l)
[ ${n_wav} -ne ${n_mb} ] && echo "Warning: expected ${n_wav} data files, found ${n_mb}" && exit 1;
[ ${n_wav} -ne ${n_fr} ] && echo "Warning: expected ${n_wav} data files, found ${n_fr}" && exit 1;

# extract meta data
find -L ${src} -iname "*.wav" | sort | while read line; do
    file_name=$(basename $line | cut -f 1 -d ".")
    echo ${file_name} | awk -v "dir=${src}" '{printf "%s cat %s/%s.wav |\n", $0, dir, $0}' >> ${wav_scp} || exit 1;

    speaker=$(basename $line | cut -f 1 -d "_")
    echo ${file_name} | awk -v "speaker=$speaker" '{printf "%s %s\n", $1, speaker}' >> ${utt2spk} || exit 1;
done

for lang in mb fr; do
    touch ${dst}/${lang}.tc.tok
    find -L ${src} -iname "*.${lang}" | sort | while read line; do
        if [ ${lang} = fr ]; then
            normalize-punctuation.perl -l ${lang} < ${line} > tokenizer.perl -l ${lang} >> ${dst}/${lang}.tc.tok
        else
            normalize-punctuation.perl -l ${lang} < ${line} > tokenizer.perl -l ${lang} >> ${dst}/${lang}.tc.tok
        fi
    done

    touch ${dst}/${lang}.lc.tok
    find -L ${src} -iname "*.${lang}.cleaned" | sort | while read line; do
        if [ ${lang} = fr ]; then
            normalize-punctuation.perl -l ${lang} < ${line} > tokenizer.perl -l ${lang} >> ${dst}/${lang}.lc.tok
        else
            normalize-punctuation.perl -l ${lang} < ${line} > tokenizer.perl -l ${lang} >> ${dst}/${lang}.lc.tok
        fi
    done

    touch ${dst}/${lang}.lc.rm.tok
    if [ ${lang} = fr ]; then
        find -L ${src} -iname "*.${lang}.cleaned.noPunct" | sort | while read line; do
            normalize-punctuation.perl -l ${lang} < ${line} > tokenizer.perl -l ${lang} >> ${dst}/${lang}.lc.rm.tok
        done
    else
        find -L ${src} -iname "*.${lang}.cleaned.split" | sort | while read line; do
            normalize-punctuation.perl -l ${lang} < ${line} > tokenizer.perl -l ${lang} >> ${dst}/${lang}.lc.rm.tok
        done
    fi

    paste -d " " <(awk '{print $1}' ${wav_scp}) <(cat ${dst}/${lang}.tc.tok) > ${dst}/text.tc.${lang}
    paste -d " " <(awk '{print $1}' ${wav_scp}) <(cat ${dst}/${lang}.lc.tok) > ${dst}/text.lc.${lang}
    paste -d " " <(awk '{print $1}' ${wav_scp}) <(cat ${dst}/${lang}.lc.rm.tok) > ${dst}/text.lc.rm.${lang}

    # save original punctuation
    detokenizer.perl -l ${lang} < ${dst}/${lang}.tc.tok | text2token.py -s 0 -n 1 | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.${lang}
    detokenizer.perl -l ${lang} < ${dst}/${lang}.lc.tok | text2token.py -s 0 -n 1 | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.clean.${lang}
done

spk2utt=${dst}/spk2utt
utils/utt2spk_to_spk2utt.pl <${utt2spk} >${spk2utt} || exit 1

utils/fix_data_dir.sh --utt_extra_files "text.tc.mb text.tc.fr \
                                         text.lc.mb text.lc.fr \
                                         text.lc.rm.mb text.lc.rm.fr" ${dst} || exit 1;
utils/validate_data_dir.sh --no-feats --no-text ${dst} || exit 1;


# error check
for f in text.tc.mb text.tc.fr; do
    ntrans=$(wc -l <${dst}/${f})
    nutt2spk=$(wc -l <${utt2spk})
    ! [ "${ntrans}" -eq "${nutt2spk}" ] && \
        echo "Inconsistent #transcripts(${ntrans}) and #utt2spk(${nutt2spk})" && exit 1;
done


# Copy stuff into its final locations [this has been moved from the format_data script]
mkdir -p data/${set}
for f in spk2utt utt2spk wav.scp; do
    cp ${dst}/${f} data/${set}/${f}
done
# mb
cp ${dst}/text.tc.mb data/${set}/text.tc.mb
cp ${dst}/text.lc.mb data/${set}/text.lc.mb
cp ${dst}/text.lc.rm.mb data/${set}/text.lc.rm.mb
# fr
cp ${dst}/text.tc.fr data/${set}/text.tc.fr
cp ${dst}/text.lc.fr data/${set}/text.lc.fr
cp ${dst}/text.lc.rm.fr data/${set}/text.lc.rm.fr


echo "$0: successfully prepared data in ${dst}"
exit 0;
