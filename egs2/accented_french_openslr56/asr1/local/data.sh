#!/bin/bash

stage=4

# devtest
if [ ${stage} -le 1 ] ; then
    # test set : 
    FILE=downloads/African_Accented_French/transcripts/devtest/ca16_read/new_conditioned.txt
    DEV_TEST_FOLDER=downloads/African_Accented_French/speech/devtest/ca16

    # .split('_') starts at 1, put in aux file, aux is the folder
    cut -d '_' -f 3 "$FILE" > aux
    cut -d ' ' -f 1 "$FILE" > uttid

    # take everything in aux $0, add that before to the $0 -> aux2
    awk '{print "downloads/African_Accented_French/speech/devtest/ca16/"$0}' aux > aux2

    # aux2/uttid -> aux3
    paste -d "/"  aux2 uttid > aux3
    awk '{print $0".wav"}' aux3 > aux4
    paste  -d " " uttid aux4 > data/test/wav.scp

    cp downloads/African_Accented_French/transcripts/devtest/ca16_read/new_conditioned.txt data/test/text
    ## utt : 

    # identity function
    paste  -d " " uttid uttid  > data/test/utt2spk
    utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

    ./utils/fix_data_dir.sh data/test/

    rm aux aux2 aux3 aux4 uttid
fi

# train
if [ ${stage} -le 2 ]; then
    # train set ca16_conv: 
    FILE=downloads/African_Accented_French/transcripts/train/ca16_conv/new_transcripts.txt

    # .split('_') starts at 1, put in aux file, aux is the folder
    cut -d '_' -f 3 "$FILE" > aux5
    cut -d ' ' -f 1 "$FILE" > uttid1
    cut -c -32 uttid1 > uttid2
    cut -d ' ' -f 2- "$FILE" > aux10

    paste -d ' ' uttid2 aux10 > auxtext1

    # take everything in aux $0, add that before to the $0 -> aux2
    awk '{print "downloads/African_Accented_French/speech/train/ca16/"$0}' aux5 > aux6

    # aux2/uttid -> aux3
    paste -d "/"  aux6 uttid2 > aux7
    awk '{print $0".wav"}' aux7 > aux8
    paste  -d " " uttid2 aux8 > auxwav1

    # identity function
    paste  -d " " uttid2 uttid2  > auxutt1

    # train set ca16_conv: 
    FILE=downloads/African_Accented_French/transcripts/train/ca16_read/new_conditioned.txt

    # .split('_') starts at 1, put in aux file, aux is the folder
    cut -d '_' -f 3 "$FILE" > aux5
    cut -d ' ' -f 1 "$FILE" > uttid1

    # take everything in aux $0, add that before to the $0 -> aux2
    awk '{print "downloads/African_Accented_French/speech/train/ca16/"$0}' aux5 > aux6

    # aux2/uttid -> aux3
    paste -d "/"  aux6 uttid1 > aux7
    awk '{print $0".wav"}' aux7 > aux8
    paste  -d " " uttid1 aux8 > auxwav2

    # identity function
    paste  -d " " uttid1 uttid1 > auxutt2

    # train yaounde read
    FILE=downloads/African_Accented_French/transcripts/train/yaounde/fn_read_text.txt

    # .split('_') starts at 1, put in aux file, aux is the folder
    cut -d '/' -f 9 "$FILE" > aux5
    cut -d ' ' -f 1 aux5 > aux6

    cut -d '/' -f 8-9 "$FILE" > aux9
    cut -d ' ' -f 1 aux9 > aux10

    cut -c -14 aux6 > aux7
    awk '{print "read-"$0}' aux7 > uttid3

    cut -d ' ' -f 2- "$FILE" > aux8
    paste -d ' ' uttid3 aux8 > auxtext3

    awk '{print "downloads/African_Accented_French/speech/train/yaounde/read/"$0}' aux10 > aux11
    paste -d ' ' uttid3 aux11 > auxwav3

    paste -d ' ' uttid3 uttid3 > auxutt3

    # train yaounde answers
    FILE=downloads/African_Accented_French/transcripts/train/yaounde/fn_answers_text.txt

    # .split('_') starts at 1, put in aux file, aux is the folder
    cut -d '/' -f 8 "$FILE" > aux5
    cut -d ' ' -f 1 aux5 > aux6

    cut -d '/' -f 7-8 "$FILE" > aux9
    cut -d ' ' -f 1 aux9 > aux10

    cut -c -13 aux6 > aux7
    awk '{print "answers-"$0}' aux7 > uttid4

    cut -d ' ' -f 2- "$FILE" > aux8
    paste -d ' ' uttid4 aux8 > auxtext4

    awk '{print "downloads/African_Accented_French/speech/train/yaounde/answers/"$0}' aux10 > aux11
    paste -d ' ' uttid4 aux11 > auxwav4

    paste -d ' ' uttid4 uttid4 > auxutt4

    # cat everything
    cat auxtext1 downloads/African_Accented_French/transcripts/train/ca16_read/new_conditioned.txt auxtext3 auxtext4 > data/train/text
    cat auxwav1 auxwav2 auxwav3 auxwav4 > data/train/wav.scp
    cat auxutt1 auxutt2 auxutt3 auxutt4 > data/train/utt2spk
    utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt

    ./utils/fix_data_dir.sh data/train/

    rm aux5 aux6 aux7 aux8 aux9 aux10 aux11 auxtext1 auxtext3 auxtext4 auxwav1 auxwav2 auxwav3 auxwav4 auxutt1 auxutt2 auxutt3 auxutt4 uttid1 uttid2 uttid3 uttid4
fi

# dev
if [ ${stage} -le 3 ]; then
    FILE=downloads/African_Accented_French/transcripts/dev/niger_west_african_fr/transcripts.txt

    cut -d '/' -f 3 "$FILE" > aux5
    cut -d ' ' -f 1 aux5 > aux6

    cut -d '/' -f 2-3 "$FILE" > aux9 
    cut -d ' ' -f 1 aux9 > aux10

    cut -c -16 aux6 > uttid

    cut -d ' ' -f 2- "$FILE" > aux8
    paste -d ' ' uttid aux8 > data/dev/text 

    awk '{print "downloads/African_Accented_French/speech/dev/niger_west_african_fr/"$0}' aux10 > aux11
    paste -d ' ' uttid aux11 > data/dev/wav.scp 

    paste -d ' ' uttid uttid > data/dev/utt2spk

    utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt

    ./utils/fix_data_dir.sh data/dev/

    rm aux5 aux6 aux9 aux10 uttid aux8 aux11
fi

# test; normalization of the test set is done in normalize_test.py
if [ ${stage} -le 4 ]; then
    FILE=downloads/African_Accented_French/transcripts/test/ca16/new_prompts.txt

    cp "$FILE" data/test/text

    cut -d ' ' -f 1 "$FILE" > aux5
    cut -d '_' -f 1-3 aux5 > aux6

    awk '{print "downloads/African_Accented_French/speech/test/ca16/"$0"/"}' aux6 > aux7
    paste -d '' aux7 aux5 > aux8
    awk '{print $0".wav"}' aux8 > aux9
    paste -d ' ' aux5 aux9 > data/test/wav.scp

    paste -d ' ' aux5 aux5 > data/test/utt2spk

    utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

    ./utils/fix_data_dir.sh data/test/

    rm aux5 aux6 aux7 aux8 aux9

fi