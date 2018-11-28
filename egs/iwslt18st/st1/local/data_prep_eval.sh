#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <src-dir>"
  echo "e.g.: $0 /export/corpora4/IWSLT/iwslt-corpus"
  exit 1
fi


part=tst2018
src=$1
dst=data/local/${part}

wav_dir=${src}/wavs
yml=${src}/IWSLT.TED.${part}.en-de.yaml

mkdir -p ${dst} || exit 1;

[ ! -d ${wav_dir} ] && echo "$0: no such directory $wav_dir" && exit 1;
[ ! -f ${yml} ] && echo "$0: expected file $yml to exist" && exit 1;

wav_scp=${dst}/wav.scp; [[ -f "$wav_scp" ]] && rm ${wav_scp}
utt2spk=${dst}/utt2spk; [[ -f "$utt2spk" ]] && rm ${utt2spk}
spk2utt=${dst}/spk2utt; [[ -f "$spk2utt" ]] && rm ${spk2utt}
segments=${dst}/segments; [[ -f "$segments" ]] && rm ${segments}

# downsample tst2018.en.lecture0001.wav from 48k to 16k
if [ ! -f ${wav_dir}/tst2018.en.lecture0001_48k.wav ]; then
  mv ${wav_dir}/tst2018.en.lecture0001.wav ${wav_dir}/tst2018.en.lecture0001_48k.wav
  sox ${wav_dir}/tst2018.en.lecture0001_48k.wav -r 16000 ${wav_dir}/tst2018.en.lecture0001.wav
fi

# downsample tst2018.en.lecture0004.wav from22.05k to 16k
# and convert from 2ch to 1ch
if [ ! -f ${wav_dir}/tst2018.en.lecture0004_22k_2ch.wav ]; then
  mv ${wav_dir}/tst2018.en.lecture0004.wav ${wav_dir}/tst2018.en.lecture0004_22k_2ch.wav
  sox ${wav_dir}/tst2018.en.lecture0004_22k_2ch.wav -r 16000 ${wav_dir}/tst2018.en.lecture0004_16k_2ch.wav
  sox ${wav_dir}/tst2018.en.lecture0004_16k_2ch.wav ${wav_dir}/tst2018.en.lecture0004.wav remix 1
  sox ${wav_dir}/tst2018.en.lecture0004_16k_2ch.wav ${wav_dir}/tst2018.en.lecture0004_16k_right.wav remix 2
fi


# (1c) Make segments files from transcript
#segments file format is: utt-id start-time end-time, e.g.:
#ted_0001_0003501_0003684 ted_0001 003.501 0003.684
cat ${yml} | awk '/./{ print $0 }' > ${dst}/.yaml0
awk '{
    wav=$3; offset=$4; duration=$5;
    gsub(",","",wav); gsub("\"","",wav);
    gsub(",","",offset); gsub("\"","",offset); gsub("offset:","",offset);
    gsub("}","",duration); gsub("\"","",duration); gsub("duration:","",duration);
    match(wav, /tst2018.en.[a-z]+[0-9]+.wav/);
    spkid = substr(wav, RSTART, RLENGTH); gsub(".wav","",spkid);
    duration=sprintf("%.7f", duration);
    if ( duration < 0.2 ) extendt=sprintf("%.7f", (0.2-duration)/2);
    else extendt=0;
    offset=sprintf("%.7f", offset);
    startt=offset-extendt;
    endt=offset+duration+extendt;
    printf("%s_%07.0f_%07.0f %s %.2f %.2f\n", spkid, int(1000*startt+0.5), int(1000*endt+0.5), spkid, startt, endt);
}' ${dst}/.yaml0 | sort > ${dst}/segments || exit 1;
# NOTE: Extend the lengths of short utterances (< 0.2s) rather than exclude them

awk '{
    spkid=$2;
    printf("%s cat '${wav_dir}'/%s.wav |\n", spkid, spkid);
}' < ${dst}/segments | uniq | sort > ${dst}/wav.scp || exit 1;

awk '{
    segment=$1; split(segment,S,"[_]");
    spkid=S[1] "_" S[2]; print $1 " " spkid
}' ${dst}/segments | sort > ${dst}/utt2spk || exit 1;

sort ${dst}/utt2spk | utils/utt2spk_to_spk2utt.pl | sort > ${dst}/spk2utt || exit 1;


# Copy stuff intoc its final locations [this has been moved from the format_data script]
mkdir -p data/${part}_en
for f in spk2utt utt2spk wav.scp segments; do
  cp ${dst}/${f} data/${part}_en/ || exit 1;
done

mkdir -p data/${part}_de
for f in spk2utt utt2spk wav.scp segments; do
  cp ${dst}/${f} data/${part}_de/ || exit 1;
done

echo "$0: successfully prepared data in $dst"
