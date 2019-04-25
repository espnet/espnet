#!/bin/bash

# Copyright 2012  Vassil Panayotov
#           2014  Johns Hopkins University (author: Daniel Povey)
#           2017  Johns Hopkins University (author: Shinji Watanabe)
# Apache 2.0

# Makes train/test splits

. path.sh

echo "=== Starting initial JNAS data preparation ..."

. utils/parse_options.sh

if [ $# != 1 ]; then
    echo "Usage: $0 <data-directory>";
    exit 1;
fi

DATA=$1
trans=Transcription
type=KANJI
wavdir=${DATA}/WAVES_HS
ifNP=NP


# make $DATA an absolute pathname.

locdata=${PWD}/data/local
loctmp=$locdata/tmp
rm -rf $loctmp >/dev/null 2>&1
mkdir -p $locdata
mkdir -p $loctmp

# extract speakers names
# from directory names into speakers_all.txt
ls $wavdir | sort | sed -e "/@eaDir/d" > $loctmp/speakers_all.txt

# get directory names and set them into dir_all.txt

if [ ! -s $loctmp/speakers_all.txt ]; then
    echo "$0: file $loctmp/speakers_all.txt is empty"
    exit 1;
fi

logdir=exp/jnas_data_prep
mkdir -p $logdir
echo -n > $logdir/make_trans.log
for s in all; do
    echo "--- Preparing ${s}_wav.scp, ${s}_trans.txt and ${s}.utt2spk ..."

    for spkname in $(cat $loctmp/speakers_${s}.txt); do
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
	done >> ${loctmp}/${s}_wav.scp.unsorted

	for a in "${all_utt2spk_entries[@]}"; do echo $a; done >> $loctmp/${s}.utt2spk.unsorted


	if [ ! -f ${loctmp}/${s}_wav.scp.unsorted ]; then
	    echo "$0: processed no data: error: pattern ${dir}/${wavtype}/*${wavtype} might match nothing"
	    exit 1;
	fi

	# check character set, and convert to utf-8
	mkdir -p ${loctmp}/char_tmp
	CHARSET=`file -bi ${scrdir}/${spkname}_${type:0:3}.txt |awk -F "=" '{print $2}'`
	# echo -n $dir/etc/PROMPTS
	# echo " encode:$CHARSET"
	if [ "$CHARSET" != 'utf-8' ] && [ "$CHARSET" != 'us-ascii' ] ; then
	    echo "iconv -f "$CHARSET" -t UTF-8 ${scrdir}/${spkname}_${type:0:3}.txt |
	    sed 's/\r//' > ${loctmp}/char_tmp/$id.utf8"
      iconv -f "$CHARSET" -t UTF-8 ${scrdir}/${spkname}_${type:0:3}.txt |
         > ${loctmp}/char_tmp/$id.utf8
      nkf --overwrite -Lu ${loctmp}/char_tmp/$id.utf8
	else
	    cp ${scrdir}/${spkname}_${type:0:3}.txt ${loctmp}/char_tmp/$id.utf8
	    nkf --overwrite -Lu  ${loctmp}/char_tmp/$id.utf8
	fi
	
	PYTHONIOENCODING=utf-8 \
	local/make_trans.py \
	  ${loctmp}/char_tmp/$id.utf8 \
	  ${spkname}_${ifNP} \
	  "${all_wavs[@]}" \
	  2>>${logdir}/make_trans.log >> ${loctmp}/${s}_trans.txt.unsorted
    done

    # filter out the audio for which there is no proper transcript
    awk 'NR==FNR{trans[$1]; next} ($1 in trans)' FS=" " \
	${loctmp}/${s}_trans.txt.unsorted ${loctmp}/${s}_wav.scp.unsorted |\
	sort -k1 > ${locdata}/${s}_wav.scp

    awk 'NR==FNR{trans[$1]; next} ($1 in trans)' FS=" " \
	${loctmp}/${s}_trans.txt.unsorted $loctmp/${s}.utt2spk.unsorted |\
	sort -k1 > ${locdata}/${s}.utt2spk

    sort -k1 < ${loctmp}/${s}_trans.txt.unsorted > ${locdata}/${s}_trans.txt

    echo "--- Preparing ${s}.spk2utt ..."
    cat $locdata/${s}_trans.txt |\
	cut -f1 -d' ' |\
	  awk 'BEGIN {FS="_"}
        {names[$1]=names[$1] " " $0;}
        END {for (k in names) {print k, names[k];}}' | sort -k1 > $locdata/${s}.spk2utt
done;

trans_err=$(wc -l <${logdir}/make_trans.log)
if [ "${trans_err}" -ge 1 ]; then
    echo -n "$trans_err errors detected in the transcripts."
    echo " Check ${logdir}/make_trans.log for details!"
fi

echo "*** Initial JNAS data preparation finished!"
