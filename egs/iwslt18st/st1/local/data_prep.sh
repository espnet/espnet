#!/bin/bash

# Copyright 2018  Hirofumi Inaguma
#           2018  Kyoto Univerity (author: Hirofumi Inaguma)
# Apache 2.0

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <src-dir> <part>"
  echo "e.g.: $0 /export/corpora4/IWSLT/iwslt-corpus dev2010"
  exit 1
fi

src=$1
part=$2
dst=data/local/$2

wav_dir=$src/wav
trans_dir=$src/parallel
yml=$trans_dir/$part.yaml
en=$trans_dir/$part.en
de=$trans_dir/$part.de

mkdir -p $dst || exit 1;

[ ! -d $wav_dir ] && echo "$0: no such directory $wav_dir" && exit 1;
[ ! -d $trans_dir ] && echo "$0: no such directory $trans_dir" && exit 1;
[ ! -f $yml ] && echo "$0: expected file $yml to exist" && exit 1;
[ ! -f $en ] && echo "$0: expected file $en to exist" && exit 1;
[ ! -f $de ] && echo "$0: expected file $de to exist" && exit 1;


wav_scp=$dst/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
trans_en=$dst/text_en; [[ -f "$trans_en" ]] && rm $trans_en
trans_de=$dst/text_de; [[ -f "$trans_de" ]] && rm $trans_de
utt2spk=$dst/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk
spk2utt=$dst/spk2utt; [[ -f "$spk2utt" ]] && rm $spk2utt
segments=$dst/segments; [[ -f "$segments" ]] && rm $segments

n=`cat $yml | grep duration | wc -l`
n_en=`cat $en | wc -l`
n_de=`cat $de | wc -l`
[ $n -ne $n_en ] && echo "Warning: expected $n data data files, found $n_en" && exit 1;
[ $n -ne $n_de ] && echo "Warning: expected $n data data files, found $n_de" && exit 1;


# (1a) Transcriptions preparation
# make basic transcription file (add segments info)
cat $yml > ${dst}/.yaml0
cat ${dst}/.yaml0 | grep duration > ${dst}/.yaml1
awk '{
    duration=$3; offset=$5; spkid=$7;
    gsub(",","",duration);
    gsub(",","",offset);
    gsub(",","",spkid);
    gsub("spk.","",spkid);
    duration=sprintf("%.6f", duration);
    offset=sprintf("%.6f", offset);
    startt=offset;
    endt=offset+duration;
    printf("ted_%04d_%07.0f_%07.0f\n", spkid, int(100*startt+0.5), int(100*endt+0.5));
}' ${dst}/.yaml1 > ${dst}/.yaml2
cat $en > ${dst}/.en0
cat $de > ${dst}/.de0

# Text normalization
local/fix_trans.py ${dst}/.en0 > ${dst}/.en1
local/fix_trans.py ${dst}/.de0 > ${dst}/.de1

# error check
n=`cat ${dst}/.yaml2 | wc -l`
n_en=`cat ${dst}/.en1 | wc -l`
n_de=`cat ${dst}/.de1 | wc -l`
[ $n -ne $n_en ] && echo "Warning: expected $n data data files, found $n_en" && exit 1;
[ $n -ne $n_de ] && echo "Warning: expected $n data data files, found $n_de" && exit 1;

paste --delimiters " " ${dst}/.yaml2 ${dst}/.en1 | awk '{
  print tolower($0) }' | sort > $dst/text_en
paste --delimiters " " ${dst}/.yaml2 ${dst}/.de1 | awk '{
  if (length($0) > 25) print tolower($0) }' | sort > $dst/text_de
# NOTE: empty utterances are includes in original German transcripts


# (1c) Make segments files from transcript
#segments file format is: utt-id start-time end-time, e.g.:
#ted_0001_0003501_0003684 ted_0001 003.501 0003.684
awk '{
    segment=$1; split(segment,S,"[_]");
    spkid=S[1] "_" S[2]; startf=S[3]; endf=S[4];
    print segment " " spkid " " startf/1000 " " endf/1000
}' < $dst/text_de | sort > $dst/segments

awk '{
    segment=$1; split(segment,S,"[_]");
    spkid=S[1] "_" S[2];
    printf("%s cat '$wav_dir'/%s_%d.wav |\n", spkid, S[1], S[2]);
}' < $dst/text_de | uniq | sort > $dst/wav.scp || exit 1;

awk '{
    segment=$1; split(segment,S,"[_]");
    spkid=S[1] "_" S[2]; print $1 " " spkid
}' $dst/segments | sort > $dst/utt2spk || exit 1;

sort $dst/utt2spk | utils/utt2spk_to_spk2utt.pl | sort > $dst/spk2utt || exit 1;


# Match the number of utterances between EN and DE (reduce EN utterances)
oldnum=`wc -l $dst/text_en | awk '{print $1}'`
utils/filter_scp.pl $dst/utt2spk < $dst/text_en > $dst/text_en_tmp
newnum=`wc -l $dst/text_en_tmp | awk '{print $1}'`
echo "change from $oldnum to $newnum"
rm $dst/text_en
mv $dst/text_en_tmp $dst/text_en

# error check
n_en=`cat ${dst}/text_en | wc -l`
n_de=`cat ${dst}/text_de | wc -l`
[ $n_en -ne $n_de ] && echo "Warning: expected $n_en data data files, found $n_de" && exit 1;


# Copy stuff intoc its final locations [this has been moved from the format_data script]
mkdir -p data/${part}_en
for f in spk2utt utt2spk wav.scp segments; do
  cp data/local/$part/$f data/${part}_en/ || exit 1;
done
cp $dst/text_en data/${part}_en/text || exit 1;

mkdir -p data/${part}_de
for f in spk2utt utt2spk wav.scp segments; do
  cp data/local/$part/$f data/${part}_de/ || exit 1;
done
cp $dst/text_de data/${part}_de/text || exit 1;

echo "$0: successfully prepared data in $dst"
