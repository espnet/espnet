#!/bin/bash

# Copyright 2012  Vassil Panayotov
#           2014  Johns Hopkins University (author: Daniel Povey)
#           2017  Johns Hopkins University (author: Shinji Watanabe)
# Apache 2.0

# Makes train/test splits

. ./path.sh

if [ $# != 1 ]; then
    echo "Usage: $0 <data-directory>";
    exit 1;
fi

DATA=$1  # database root directory

echo "=== Starting initial JNAS data preparation ..."

wavdir=${DATA}/WAVES_HS
trans=Transcription  # transcription dir name
type=KANJI  # transcription type
ifNP=NP
locdata=${PWD}/data
loctmp=$locdata/all/tmp

rm -rf $loctmp >/dev/null 2>&1
mkdir -p ${locdata}/all/tmp

# extract speakers names
# from directory names into speakers_all.txt
ls $wavdir | sort | grep -e ^M -e ^F > $loctmp/speakers_all.txt

# get directory names and set them into dir_all.txt
if [ ! -s $loctmp/speakers_all.txt ]; then
    echo "$0: file $loctmp/speakers_all.txt is empty"
    exit 1;
fi

logdir=exp/jnas_data_prep
mkdir -p $logdir
echo -n > $logdir/make_trans.log

echo "--- Preparing all/wav.scp, all/trans.txt and all/utt2spk ..."

for spkname in $(cat $loctmp/speakers_all.txt); do
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

    all_wavs=()
    all_utt2spk_entries=()
    for w in ${spkwav_dir}/*${wavtype}; do
        bw=`basename $w`
        wavname=${bw%.$wavtype}
        all_wavs+=("${wavname:0:-3}")
        id="${spkname}_${ifNP}_${wavname:0:-3}"
        if [ ! -s $w ]; then
            echo "$w is zero-size - skipping ..." 1>&2
            continue
        fi
        echo "$id $w"
        all_utt2spk_entries+=("$id $spkname")
    done >> ${loctmp}/all_wav.scp.unsorted

    for a in "${all_utt2spk_entries[@]}"; do echo $a; done >> $loctmp/all.utt2spk.unsorted

    if [ ! -f ${loctmp}/all_wav.scp.unsorted ]; then
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

    PYTHONIOENCODING=utf-8 local/make_trans.py \
        ${loctmp}/char_tmp/$id.utf8 \
        ${spkname}_${ifNP} \
        "${all_wavs[@]}" \
        2>>${logdir}/make_trans.log >> ${loctmp}/all_trans.txt.unsorted
done

# filter out the audio for which there is no proper transcript
awk 'NR==FNR{trans[$1]; next} ($1 in trans)' FS=" " \
${loctmp}/all_trans.txt.unsorted ${loctmp}/all_wav.scp.unsorted |\
sort -k1 > ${locdata}/all/wav.scp

awk 'NR==FNR{trans[$1]; next} ($1 in trans)' FS=" " \
${loctmp}/all_trans.txt.unsorted $loctmp/all.utt2spk.unsorted |\
sort -k1 > ${locdata}/all/utt2spk

sort -k1 < ${loctmp}/all_trans.txt.unsorted > ${locdata}/all/text

echo "--- Preparing all.spk2utt ..."
cat $locdata/all/text |\
cut -f1 -d' ' |\
  awk 'BEGIN {FS="_"}
    {names[$1]=names[$1] " " $0;}
    END {for (k in names) {print k, names[k];}}' | sort -k1 > $locdata/all/spk2utt

trans_err=$(wc -l <${logdir}/make_trans.log)
if [ "${trans_err}" -ge 1 ]; then
    echo -n "$trans_err errors detected in the transcripts."
    echo " Check ${logdir}/make_trans.log for details!"
fi

# check the structure of perepraed data directory
utils/fix_data_dir.sh ${locdata}/all
rm -rf ${loctmp}

echo "*** Initial JNAS data preparation finished!"
