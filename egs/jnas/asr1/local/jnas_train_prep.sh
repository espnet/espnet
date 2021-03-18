#!/usr/bin/env bash

# Copyright 2019 Nagoya University (Someki Masao) and Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Prepare JNAS dataset

. ./path.sh || exit 1;

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <data-directory> <speaker_text> [<trans_type>]";
    exit 1;
fi

DATA=$1  # database root directory
speaker_list=$2
trans_type=${3:-kanji}

echo "=== Starting initial JNAS training_data preparation ..."

wavdir=${DATA}/WAVES_HS
trans=Transcription  # transcription dir name
type=KANJI  # transcription type
ifNP=NP
locdata=${PWD}/data
loctmp=$locdata/train/tmp
rm -rf $loctmp >/dev/null 2>&1
mkdir -p ${locdata}/train/tmp

# extract speakers names

logdir=exp/train_prep
mkdir -p $logdir
echo -n > $logdir/make_trans.log

echo "--- Preparing train/wav.scp, train/trans.txt and train/utt2spk ..."

for spkname in $(cat ${speaker_list}); do
    scrdir=${DATA}/${trans}/${type}/${ifNP}
    spkwav_dir=${wavdir}/${spkname}/${ifNP}

    if [ ! -f ${scrdir}/${spkname}_${type:0:3}.txt ]; then
        echo "No ${spkname}_${type:0:3}.txt file exists in $scrdir - skipping the dir ..." \
         >> $logdir/make_trans.log
    fi

    if ls ${spkwav_dir}/*.wav > /dev/null 2>&1; then
        wavtype=wav
    else
        echo "No 'wav' dir in $spkwav_dir - skipping ..."
        continue
    fi

    train_wavs=()
    train_utt2spk_entries=()
    for w in ${spkwav_dir}/*${wavtype}; do
        bw=`basename $w`
        wavname=${bw%.$wavtype}
        train_wavs+=("${wavname:0:-3}")
        id="${spkname}_${ifNP}_${wavname:0:-3}"
        if [ ! -s $w ]; then
            echo "$w is zero-size - skipping ..." 1>&2
            continue
        fi
        echo "$id $w"
        train_utt2spk_entries+=("$id $spkname")
    done >> ${loctmp}/train_wav.scp.unsorted

    for a in "${train_utt2spk_entries[@]}"; do echo $a; done >> $loctmp/train_utt2spk.unsorted

    if [ ! -f ${loctmp}/train_wav.scp.unsorted ]; then
        echo "$0: processed no data: error: pattern ${dir}/${wavtype}/*${wavtype} might match nothing"
        exit 1;
    fi

    # check character set, and convert to utf-8
    mkdir -p ${loctmp}/char_tmp
    CHARSET=`file -bi ${scrdir}/${spkname}_${type:0:3}.txt |awk -F "=" '{print $2}'`
    if [ "$CHARSET" != 'utf-8' ] && [ "$CHARSET" != 'us-ascii' ] ; then
        echo "iconv -f "$CHARSET" -t UTF-8 ${scrdir}/${spkname}_${type:0:3}.txt |
        sed 's/\r//' > ${loctmp}/char_tmp/$id.utf8"
      iconv -f "$CHARSET" -t UTF-8 ${scrdir}/${spkname}_${type:0:3}.txt |\
          > ${loctmp}/char_tmp/$id.utf8
      nkf --overwrite -Lu ${loctmp}/char_tmp/$id.utf8
    else
        cp ${scrdir}/${spkname}_${type:0:3}.txt ${loctmp}/char_tmp/$id.utf8
        nkf --overwrite -Lu  ${loctmp}/char_tmp/$id.utf8
    fi

    local/make_train_trans.py \
        ${loctmp}/char_tmp/$id.utf8 \
        ${spkname}_${ifNP} \
        "${train_wavs[@]}" \
        2>>${logdir}/make_trans.log >> ${loctmp}/train_trans.txt.unsorted
done

# filter out the audio for which there is no proper transcript
awk 'NR==FNR{trans[$1]; next} ($1 in trans)' FS=" " \
${loctmp}/train_trans.txt.unsorted ${loctmp}/train_wav.scp.unsorted |\
sort -k1 > ${locdata}/train/wav.scp

awk 'NR==FNR{trans[$1]; next} ($1 in trans)' FS=" " \
${loctmp}/train_trans.txt.unsorted $loctmp/train_utt2spk.unsorted |\
sort -k1 > ${locdata}/train/utt2spk

sort -k1 < ${loctmp}/train_trans.txt.unsorted > ${locdata}/train/text.tmp

# remove spaces
paste -d " " <(cut -f 1 -d" " ${locdata}/train/text.tmp) <(cut -f 2- -d" " ${locdata}/train/text.tmp | tr -d " ") > ${locdata}/train/text
rm ${locdata}/train/text.tmp

echo "--- Preparing train/spk2utt ..."
cat $locdata/train/text |\
cut -f1 -d' ' |\
  awk 'BEGIN {FS="_"}
    {names[$1]=names[$1] " " $0;}
    END {for (k in names) {print k, names[k];}}' | sort -k1 > $locdata/train/spk2utt

trans_err=$(wc -l <${logdir}/make_trans.log)
if [ "${trans_err}" -ge 1 ]; then
    echo -n "$trans_err errors detected in the transcripts."
    echo " Check ${logdir}/make_trans.log for details!"
fi

# check the structure of perepraed data directory
utils/fix_data_dir.sh ${locdata}/train
rm -rf ${loctmp}

# convert text type (only for tts)
if [ ${trans_type} != "kanji" ]; then
    mv ${locdata}/train/text ${locdata}/train/rawtext
    local/clean_text.py ${locdata}/train/rawtext ${locdata}/train/text ${trans_type}
    rm ${locdata}/train/rawtext
fi
echo "*** Initial JNAS training_data preparation finished!"
