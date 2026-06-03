#!/usr/bin/env bash

# DIRHA DATA PREPARATION   (Author: Mirco Ravanelli)

DATA_FOLDER=$1
MAT_TYPE=$2


mkdir data
mkdir data/$MAT_TYPE


find $DATA_FOLDER -name \*.txt -exec bash -c \
'echo {};cat {}' \; | sed 's/\.txt//' | awk '{filename = match($0, "/"); if (filename>0){gsub("/"," ",$0); if (NR!=1) {printf "\n"} {printf "%s%s%s-%s ",$(NF-3),$(NF-2),$(NF-1),$(NF)} } \
else {if ($3 != "sil") {gsub("\\([2-4]\\)","",$3);gsub("_tr[1-9]_","",$3);$1="";$2="";sub("  ", "");printf "%s ",toupper($0)}} }' | sort -k1,1 > data/$MAT_TYPE/text


find $DATA_FOLDER -name \*.wav | sed 's/\.wav//' | \
awk -F "/" '{if (NR!=1) {printf "\n"} {printf "%s%s%s-%s %s.wav",$(NF-3),$(NF-2),$(NF-1),$(NF), $0} }'  | sort -k1,1 > data/$MAT_TYPE/wav.scp


find $DATA_FOLDER -name \*.wav | sed 's/\.wav//' | \
awk -F "/" '{if (NR!=1) {printf "\n"} {printf "%s%s%s-%s %s%s%s",$(NF-3),$(NF-2),$(NF-1),$(NF), $(NF-3),$(NF-2),$(NF-1)} }' | sort -k1,1 > data/$MAT_TYPE/utt2spk


find $DATA_FOLDER -name \*.wav | sed 's/\.wav//' | \
awk -F "/" '{gender = match($0, "/Male/"); if (gender>0) {if (NR!=1) {printf "\n"} printf "%s%s%s m",$(NF-3),$(NF-2),$(NF-1)} \
else {if (NR!=1) {printf "\n"} printf "%s%s%s f",$(NF-3),$(NF-2),$(NF-1)} }' | sort -k1,1 | uniq > data/$MAT_TYPE/spk2gender

utils/utt2spk_to_spk2utt.pl data/$MAT_TYPE/utt2spk > data/$MAT_TYPE/spk2utt


# Prepare STM file for sclite:
wav-to-duration scp:data/$MAT_TYPE/wav.scp ark,t:data/$MAT_TYPE/dur.ark || exit 1
  awk -v dur=data/$MAT_TYPE/dur.ark \
  'BEGIN{
     while(getline < dur) { durH[$1]=$2; }
     print ";; LABEL \"O\" \"Overall\" \"Overall\"";
     print ";; LABEL \"F\" \"Female\" \"Female speakers\"";
     print ";; LABEL \"M\" \"Male\" \"Male speakers\"";
   }
   { wav=$1; spk=gensub(/-.*/,"",1,wav); $1=""; ref=$0;
     gender = match(spk, "Female"); if (gender>0) { printf("%s 1 %s 0.0 %f <O,F> %s\n", wav, spk, durH[wav], ref);} else { printf("%s 1 %s 0.0 %f <O,M> %s\n", wav, spk, durH[wav], ref);};

   }
  ' data/$MAT_TYPE/text > data/$MAT_TYPE/stm || exit 1

  # Create dummy GLM file for sclite:
  echo ';; empty.glm
  [FAKE]     =>  %HESITATION     / [ ] __ [ ] ;; hesitation token
  ' > data/$MAT_TYPE/glm
