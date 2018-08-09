#!/bin/bash

# Copyright 2018  Hirofumi Inaguma
#           2018  Kyoto Univerity (author: Hirofumi Inaguma)
# Apache 2.0

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <src-dir> <dst-dir>"
  echo "e.g.: $0 /export/corpora4/IWSLT/iwslt-corpus data/dev2010"
  exit 1
fi

src=$1
dst=$2
part=$(basename $dst)

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
trans_en=$dst/text; [[ -f "$trans_en" ]] && rm $trans_en
trans_de=$dst/text_de; [[ -f "$trans_de" ]] && rm $trans_de
utt2spk=$dst/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk

n=`cat $yml | grep duration | wc -l`
[ $n -ne `cat $en | wc -l` ] && echo "Warning: expected 171121 data data files, found $n" && exit 1;
[ $n -ne `cat $de | wc -l` ] && echo "Warning: expected 171121 data data files, found $n" && exit 1;


# (1a) Transcriptions preparation
# make basic transcription file (add segments info)

##e.g A01F0055_0172 00380.213 00385.951 => A01F0055_0380213_0385951
cat $yml | grep duration > .tmp
count=0
awk '{
    duration=$3; offset=$5; spkid=$7;
    gsub(",","",duration);
    gsub(",","",offset);
    gsub(",","",spkid);
    gsub("spk.","",spkid);
    duration=sprintf("%.3f", duration);
    offset=sprintf("%.3f", offset);
    printf("ted_%04d_%06.0f_%06.0f\n",
           spkid, int(100*offset+0.5), int(100*offset+100*duration+0.5));
}' .tmp > .tmp2
rm .tmp

paste --delimiters " " .tmp2 $en | awk '{ print tolower($0) }' | sort > $dst/text
paste --delimiters " " .tmp2 $de | awk '{ print tolower($0) }' | sort > $dst/text_de
rm .tmp2


# (1c) Make segments files from transcript
#segments file format is: utt-id start-time end-time, e.g.:
#A01F0055_0380213_0385951 => A01F0055_0380213_0385951 A01F0055 00380.213 00385.951
awk '{
    segment=$1; split(segment,S,"[_]");
    spkid=S[1] "_" S[2]; startf=S[3]; endf=S[4];
    print segment " " spkid " " startf/1000 " " endf/1000
}' < $dst/text | sort > $dst/segments

awk '{
    segment=$1; split(segment,S,"[_]");
    spkid=S[1] "_" S[2];
    printf("%s cat '$wav_dir'/%s_%d.wav |\n", spkid, S[1], S[2]);
}' < $dst/text | uniq | sort > $dst/wav.scp || exit 1;

awk '{
    segment=$1; split(segment,S,"[_]");
    spkid=S[1] "_" S[2]; print $1 " " spkid
}' $dst/segments | sort > $dst/utt2spk || exit 1;

sort $dst/utt2spk | utils/utt2spk_to_spk2utt.pl | sort > $dst/spk2utt || exit 1;

# Copy stuff into its final locations [this has been moved from the format_data script]
mkdir -p data/$part
for f in spk2utt utt2spk wav.scp text text_de segments; do
  cp data/local/$part/$f data/$part/ || exit 1;
done

echo "$0: successfully prepared data in $dst"
