#!/bin/bash



# test set : 

FILE=downloads/African_Accented_French/transcripts/devtest/ca16_read/conditioned.txt
DEV_TEST_FOLDER=downloads/African_Accented_French/speech/devtest/ca16

cut -d '_' -f 3 "$FILE" > aux
cut -d ' ' -f 1 "$FILE" > uttid


awk '{print "downloads/African_Accented_French/speech/devtest/ca16/"$0}' aux > aux2
#awk '{print $0"lala"}' aux2 > aux3

paste -d "/"  aux2 uttid > aux3
awk '{print $0".wav"}' aux3 > aux4
paste  -d " " uttid aux4 > data/test/wav.scp

cp downloads/African_Accented_French/transcripts/devtest/ca16_read/conditioned.txt data/test/text
## utt : 

paste  -d " " uttid uttid  > data/test/utt2spk
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

./utils/fix_data_dir.sh data/test/

#rm aux aux2 aux3 aux4 uttid