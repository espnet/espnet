#!/usr/bin/env bash

# Copyright 2019 Nagoya University (Someki Masao) and Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Prepare JNAS dataset

. ./path.sh || exit 1;

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <data-directory> <speaker_text> [<trans_type>]";
    exit 1;
fi

source_root=$1  # database root directory
EVAL_PATH=$2
trans_type=${3:-kanji}

DATA=${source_root}/${EVAL_PATH}

echo "=== Starting initial JNAS eval data preparation ..."

wavdir=${DATA}/WAVES
trans=Transcription  # transcription dir name
type=KANJI  # transcription type
locdata=${PWD}/data
loctmp=$locdata/${EVAL_PATH}/tmp

rm -rf $loctmp >/dev/null 2>&1
mkdir -p ${loctmp}


logdir=exp/${EVAL_PATH}
mkdir -p $logdir
echo -n > $logdir/make_trans.log

echo "--- Preparing ${EVAL_PATH}/wav.scp, ${EVAL_PATH}/trans.txt and ${EVAL_PATH}/utt2spk ..."

for filename in $(ls $wavdir); do
    scrdir=${DATA}/${trans}/${type}
    spkwav_dir=${DATA}/WAVES/${filename}

    if [ ! -f ${scrdir}/${filename}_${type:0:3}.txt ]; then
        echo "No ${filename}_${type:0:3}.txt file exists in $scrdir - skipping the dir ..." \
         >> $logdir/make_trans.log
    fi

    if ls ${spkwav_dir}/*.wav > /dev/null 2>&1; then
        wavtype=wav
    else
        echo "No 'wav' dir in $spkwav_dir - skipping ..."
        continue
    fi

    all_wavs=()
    all_utt2spk_entries=()
    all_ids=()
    for w in ${spkwav_dir}/*${wavtype}; do
        bw=`basename $w`
        wavname=${bw%.$wavtype}
	spkname=${wavname:1:4}
        all_wavs+=("${wavname:0:-3}")
        id="${spkname}_${filename}_${wavname:0:-3}"
	all_ids+=("$id")
        if [ ! -s $w ]; then
            echo "$w is zero-size - skipping ..." 1>&2
            continue
        fi
        echo "$id $w"
        all_utt2spk_entries+=("$id $spkname")
    done >> ${loctmp}/${EVAL_PATH}_wav.scp.unsorted

    for a in "${all_utt2spk_entries[@]}"; do echo $a; done >> $loctmp/${EVAL_PATH}_utt2spk.unsorted

    if [ ! -f ${loctmp}/${EVAL_PATH}_wav.scp.unsorted ]; then
        echo "$0: processed no data: error: pattern ${filename}/*${wavtype} might match nothing"
        exit 1;
    fi

    # check character set, and convert to utf-8
    mkdir -p ${loctmp}/char_tmp
    CHARSET=`file -bi ${scrdir}/${filename}_${type:0:3}.txt |awk -F "=" '{print $2}'`
    if [ "$CHARSET" != 'utf-8' ] && [ "$CHARSET" != 'us-ascii' ] ; then
        echo "iconv -f "$CHARSET" -t UTF-8 ${scrdir}/${filename}_${type:0:3}.txt |
        sed 's/\r//' > ${loctmp}/char_tmp/$id.utf8"
      iconv -f "$CHARSET" -t UTF-8 ${scrdir}/${filename}_${type:0:3}.txt |\
      > ${loctmp}/char_tmp/$id.utf8
      nkf --overwrite -Lu ${loctmp}/char_tmp/$id.utf8
    else
        cp ${scrdir}/${filename}_${type:0:3}.txt ${loctmp}/char_tmp/$id.utf8
        nkf --overwrite -Lu  ${loctmp}/char_tmp/$id.utf8
    fi

    local/make_eval_trans.py \
        ${loctmp}/char_tmp/$id.utf8 \
        "${all_ids[@]}" \
        2>>${logdir}/make_trans.log >> ${loctmp}/${EVAL_PATH}_trans.txt.unsorted
done

# filter out the audio for which there is no proper transcript
awk 'NR==FNR{trans[$1]; next} ($1 in trans)' FS=" " \
${loctmp}/${EVAL_PATH}_trans.txt.unsorted ${loctmp}/${EVAL_PATH}_wav.scp.unsorted |\
sort -k1 > ${locdata}/${EVAL_PATH}/wav.scp


awk 'NR==FNR{trans[$1]; next} ($1 in trans)' FS=" " \
${loctmp}/${EVAL_PATH}_trans.txt.unsorted $loctmp/${EVAL_PATH}_utt2spk.unsorted |\
sort -k1 > ${locdata}/${EVAL_PATH}/utt2spk

sort -k1 < ${loctmp}/${EVAL_PATH}_trans.txt.unsorted > ${locdata}/${EVAL_PATH}/text.tmp

# remove spaces
paste -d " " <(cut -f 1 -d" " ${locdata}/${EVAL_PATH}/text.tmp) <(cut -f 2- -d" " ${locdata}/${EVAL_PATH}/text.tmp | tr -d " ") > ${locdata}/${EVAL_PATH}/text
rm ${locdata}/${EVAL_PATH}/text.tmp

echo "--- Preparing all.spk2utt ..."
cat $locdata/${EVAL_PATH}/text |\
cut -f1 -d' ' |\
  awk 'BEGIN {FS="_"}
    {names[$1]=names[$1] " " $0;}
    END {for (k in names) {print k, names[k];}}' | sort -k1 > $locdata/${EVAL_PATH}/spk2utt

trans_err=$(wc -l <${logdir}/make_trans.log)
if [ "${trans_err}" -ge 1 ]; then
    echo -n "$trans_err errors detected in the transcripts."
    echo " Check ${logdir}/make_trans.log for details!"
fi

# check the structure of perepraed data directory
utils/fix_data_dir.sh ${locdata}/${EVAL_PATH}
rm -rf ${loctmp}

# convert text type (only for tts)
if [ ${trans_type} != "kanji" ]; then
    mv ${locdata}/${EVAL_PATH}/text ${locdata}/${EVAL_PATH}/rawtext
    local/clean_text.py ${locdata}/${EVAL_PATH}/rawtext ${locdata}/${EVAL_PATH}/text ${trans_type}
    rm ${locdata}/${EVAL_PATH}/rawtext
fi
echo "*** Initial JNAS eval data preparation finished!"
