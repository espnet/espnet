#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <src-dir> <set>"
  echo "e.g.: $0 /export/corpora4/IWSLT dev2010"
  exit 1
fi


set=$2
src=$1/$set/IWSLT.$set
dst=data/local/$set

wav_dir=$src/wavs
xml_en=$src/IWSLT.TED.$set.en-de.en.xml
xml_de=$src/IWSLT.TED.$set.en-de.de.xml
if [ $set = tst2018 ]; then
  yml=$src/IWSLT.TED.$set.en-de.yaml
else
  yml=$src/test-db.yaml
fi

mkdir -p $dst

[ ! -d $wav_dir ] && echo "$0: no such directory $wav_dir" && exit 1;
if [ $set != tst2018 ]; then
  [ ! -f $xml_en ] && echo "$0: expected file $xml_en to exist" && exit 1;
  [ ! -f $xml_de ] && echo "$0: expected file $xml_de to exist" && exit 1;
fi

wav_scp=$dst/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
trans_en=$dst/text_en; [[ -f "$trans_en" ]] && rm $trans_en
trans_de=$dst/text_de; [[ -f "$trans_de" ]] && rm $trans_de
utt2spk=$dst/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk
spk2utt=$dst/spk2utt; [[ -f "$spk2utt" ]] && rm $spk2utt
segments=$dst/segments; [[ -f "$segments" ]] && rm $segments


if [ $set = tst2018 ]; then
  # downsample tst2018.en.lecture0001.wav from 48k to 16k
  if [ ! -f $wav_dir/tst2018.en.lecture0001_48k.wav ]; then
    mv $wav_dir/tst2018.en.lecture0001.wav $wav_dir/tst2018.en.lecture0001_48k.wav
    sox $wav_dir/tst2018.en.lecture0001_48k.wav -r 16000 $wav_dir/tst2018.en.lecture0001.wav
  fi

  # downsample tst2018.en.lecture0004.wav from22.05k to 16k
  # and convert from 2ch to 1ch
  if [ ! -f $wav_dir/tst2018.en.lecture0004_22k_2ch.wav ]; then
    mv $wav_dir/tst2018.en.lecture0004.wav $wav_dir/tst2018.en.lecture0004_22k_2ch.wav
    sox $wav_dir/tst2018.en.lecture0004_22k_2ch.wav -r 16000 $wav_dir/tst2018.en.lecture0004_16k_2ch.wav
    sox $wav_dir/tst2018.en.lecture0004_16k_2ch.wav $wav_dir/tst2018.en.lecture0004.wav remix 1
    sox $wav_dir/tst2018.en.lecture0004_16k_2ch.wav $wav_dir/tst2018.en.lecture0004_16k_right.wav remix 2
  fi
fi
# TODO(hirofumi): Remove this after updating download URL


# (1a) Transcriptions preparation
if [ $set != tst2018 ]; then
  # make basic transcription file (add segments info)
  python local/parse_xml.py $xml_en | sort > $dst/.en0
  python local/parse_xml.py $xml_de | sort > $dst/.de0

  # normalize punctuation
  paste -d " " <(awk '{print $1}' $dst/.en0) <(cut -f 2- -d " " $dst/.en0 | normalize-punctuation.perl -l en | tokenizer.perl -a -l en) > $dst/test_en
  paste -d " " <(awk '{print $1}' $dst/.de0) <(cut -f 2- -d " " $dst/.de0 | normalize-punctuation.perl -l de | tokenizer.perl -a -l de) > $dst/test_de
  # NOTE: case-sensitive

  # error check
  n_en=`cat $dst/test_en | wc -l`
  n_de=`cat $dst/test_de | wc -l`
  [ $n_en -ne $n_de ] && echo "Warning: expected $n_en data data files, found $n_de" && exit 1;
fi


# (1b) Segmente audio file with LIUM diarization tool
if [ $set != tst2018 ]; then
  echo "" > $src/test-db.yaml
  for f in `cat $src/FILE_ORDER`
  do
    java -jar ../../../tools/lium_spkdiarization-8.4.1.jar --fInputSpeechThr=0.0 --fInputMask=$wav_dir/$f.wav --sOutputMask=$wav_dir/$f.seg $f --saveAllStep
    # using *.s.seg for now, we live with possibly bad segmentation instead of throwing away to much stuff
    # also sort by start offset of utterance
    cat $wav_dir/$f.s.seg | grep --invert-match ";;" | sort -n -k3 | awk '{print "- {\"wav\": \"PATH/wavs/" $1 ".wav\", \"offset\":" $3/100 ", \"duration\":" ($4)/100 "}"}' >> $src/test-db.yaml
  done
  sed -i 's\PATH\'$src'\g' $src/test-db.yaml
fi
# NOTE: audio segmentaion and golden transcripts don't match here
# After finishing the training stage, hyp and ref are aligned by a RWTH tool


# (1c) Make segments files from $src/test-db.yaml
#segments file format is: utt-id start-time end-time, e.g.:
#ted_0001_0003501_0003684 ted_0001 003.501 0003.684
cat $yml | awk '/./{ print $0 }' > $dst/.yaml0
awk '{
    wav=$3; offset=$4; duration=$5;
    gsub(",","",wav); gsub("\"","",wav);
    gsub(",","",offset); gsub("\"","",offset); gsub("offset:","",offset);
    gsub("}","",duration); gsub("\"","",duration); gsub("duration:","",duration);
    match(wav, /\/[a-z0-9]+.en.[a-z]+[0-9]+.wav/);
    spkid = substr(wav, RSTART, RLENGTH); gsub(".wav","",spkid); gsub("/","",spkid);
    duration=sprintf("%.7f", duration);
    if ( duration < 0.2 ) extendt=sprintf("%.7f", (0.2-duration)/2);
    else extendt=0;
    offset=sprintf("%.7f", offset);
    startt=offset-extendt;
    endt=offset+duration+extendt;
    printf("%s_%07.0f_%07.0f %s %.2f %.2f\n", spkid, int(1000*startt+0.5), int(1000*endt+0.5), spkid, startt, endt);
}' $dst/.yaml0 | sort > $dst/segments
# NOTE: Extend the lengths of short utterances (< 0.2s) rather than exclude them

awk '{
    spkid=$2;
    printf("%s cat '$wav_dir'/%s.wav |\n", spkid, spkid);
}' < $dst/segments | uniq | sort > $dst/wav.scp

awk '{
    segment=$1; split(segment,S,"[_]");
    spkid=S[1]; print $1 " " spkid
}' $dst/segments | sort > $dst/utt2spk

sort $dst/utt2spk | utils/utt2spk_to_spk2utt.pl | sort > $dst/spk2utt


# Copy stuff intoc its final locations [this has been moved from the format_data script]
mkdir -p data/${set}_en
for f in spk2utt utt2spk wav.scp segments; do
  cp $dst/$f data/${set}_en/
done
if [ $set != tst2018 ]; then
  cp $dst/test_en data/${set}_en/text_noseg
  # NOTE: for passing utils/validate_data_dir.sh
fi

mkdir -p data/${set}_de
for f in spk2utt utt2spk wav.scp segments; do
  cp $dst/$f data/${set}_de/
done
if [ $set != tst2018 ]; then
  cp $dst/test_de data/${set}_de/text_noseg
  # NOTE: for passing utils/validate_data_dir.sh
fi

echo "$0: successfully prepared data in $dst"
