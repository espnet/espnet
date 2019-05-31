#!/bin/bash

# Copyright 2012  Vassil Panayotov
#           2014  Johns Hopkins University (author: Daniel Povey)
#           2017  Johns Hopkins University (author: Shinji Watanabe)
# Apache 2.0

# Makes train/test splits

. path.sh

echo "=== Starting initial VoxForge data preparation ..."

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <data-directory> <lang>";
    exit 1;
fi

command -v flac >/dev/null 2>&1 ||\
    { echo "FLAC decompressor needed but not found"'!' ; exit 1; }

DATA=$1
lang=$2

# make $DATA an absolute pathname.
DATA=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $DATA ${PWD}`

locdata=data/local/$lang
loctmp=$locdata/tmp
rm -rf $loctmp >/dev/null 2>&1
mkdir -p $locdata
mkdir -p $loctmp
# The "sed" expression below is quite messy because some of the directrory
# names don't follow the "speaker-YYYYMMDD-<random_3letter_suffix>" convention.
# The ";tx;d;:x" part of the expression is to filter out the directories,
# not matched by the expression
find $DATA/ -mindepth 1 -maxdepth 1 |\
    perl -ane ' s:.*/((.+)\-[0-9]{8,10}[a-z]*([_\-].*)?):$2: && print; ' | \
    sort -u > $loctmp/speakers_all.txt

wc -l $loctmp/speakers_all.txt

# expand speaker names to their respective directories
for d in $(find ${DATA}/ -mindepth 1 -maxdepth 1 -type l -or -type d); do
    basename $d
done | awk 'BEGIN {FS="-"} NR==FNR{arr[$1]; next;} ($1 in arr)' \
	   $loctmp/speakers_all.txt - | sort > $loctmp/dir_all.txt
if [ ! -s $loctmp/dir_all.txt ]; then
    echo "$0: file $loctmp/dir_all.txt is empty"
    exit 1;
fi

logdir=exp/data_prep_${lang}
mkdir -p $logdir
echo -n > $logdir/make_trans.log
for s in all; do
    echo "--- Preparing ${s}_wav.scp, ${s}_trans.txt and ${s}.utt2spk ..."

    for d in $(cat $loctmp/dir_${s}.txt); do
	spkname=`echo $d | cut -f1 -d'-'`;
	spksfx=`echo $d | cut -f2- -d'-'`; # | sed -e 's:_:\-:g'`;
	idpfx="${spkname}-${spksfx}";
	dir=${DATA}/$d

	rdm=`find $dir/etc/ -iname 'readme'`
	if [ -z $rdm ]; then
	    echo "No README file for $d - skipping this directory ..."
	    continue
	fi

	if [ ! -f ${dir}/etc/PROMPTS ]; then
	    echo "No etc/PROMPTS file exists in $dir - skipping the dir ..." \
		 >> $logdir/make_trans.log
	    continue
	fi

	if [ -d ${dir}/wav ]; then
	    wavtype=wav
	elif [ -d ${dir}/flac ]; then
	    wavtype=flac
	else
	    echo "No 'wav' or 'flac' dir in $dir - skipping ..."
	    continue
	fi

	all_wavs=()
	all_utt2spk_entries=()
	for w in ${dir}/${wavtype}/*${wavtype}; do
	    bw=`basename $w`
	    wavname=${bw%.$wavtype}
	    all_wavs+=("$wavname")
	    id="${idpfx}-${wavname}"
	    if [ ! -s $w ]; then
		echo "$w is zero-size - skipping ..." 1>&2
		continue
	    fi
	    if [ $wavtype == "wav" ]; then
		echo "$id $w"
	    else
		echo "$id flac -c -d --silent $w |"
	    fi
	    all_utt2spk_entries+=("$id $spkname")
	done >> ${loctmp}/${s}_wav.scp.unsorted

	for a in "${all_utt2spk_entries[@]}"; do echo $a; done >> $loctmp/${s}.utt2spk.unsorted


	if [ ! -f ${loctmp}/${s}_wav.scp.unsorted ]; then
	    echo "$0: processed no data: error: pattern ${dir}/${wavtype}/*${wavtype} might match nothing"
	    exit 1;
	fi

	# check character set, and convert to utf-8
	mkdir -p ${loctmp}/char_tmp
	CHARSET=`file -bi $dir/etc/PROMPTS |awk -F "=" '{print $2}'`
	# echo -n $dir/etc/PROMPTS
	# echo " encode:$CHARSET"
	if [ "$CHARSET" != 'utf-8' ] && [ "$CHARSET" != 'us-ascii' ] ; then
	    echo "iconv -f "$CHARSET" -t UTF-8 $dir/etc/PROMPTS > ${loctmp}/char_tmp/$idpfx.utf8"
	    iconv -f "$CHARSET" -t UTF-8 $dir/etc/PROMPTS  > ${loctmp}/char_tmp/$idpfx.utf8
	else
	    cp $dir/etc/PROMPTS ${loctmp}/char_tmp/$idpfx.utf8
	fi
  
	PYTHONIOENCODING=utf-8 local/make_trans.py ${loctmp}/char_tmp/$idpfx.utf8 ${idpfx} "${all_wavs[@]}" \
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
	  awk 'BEGIN {FS="-"}
        {names[$1]=names[$1] " " $0;}
        END {for (k in names) {print k, names[k];}}' | sort -k1 > $locdata/${s}.spk2utt
done;

trans_err=$(wc -l <${logdir}/make_trans.log)
if [ "${trans_err}" -ge 1 ]; then
    echo -n "$trans_err errors detected in the transcripts."
    echo " Check ${logdir}/make_trans.log for details!"
fi

echo "*** Initial VoxForge data preparation finished!"
