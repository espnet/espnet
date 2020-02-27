if [[ "$(hostname -d)" == clsp.jhu.edu ]]; then
    AISHELL=
    AN4=
    WSJ0=
    WSJ1=
    WSJCAM0=
    REVERB=
    CHIME3=
    CHIME4=
    CSJDATATOP=/export/corpora5/CSJ/USB
    CSJVER=usb  ## Set your CSJ format (dvd or usb).
                ## Usage    :
                ## Case DVD : We assume CSJ DVDs are copied in this directory with the names dvd1, dvd2,...,dvd17.
                ##            Neccesary directory is dvd3 - dvd17.
                ##            e.g. $ ls $CSJDATATOP(DVD) => 00README.txt dvd1 dvd2 ... dvd17
                ##
                ## Case USB : Neccesary directory is MORPH/SDB and WAV
                ##            e.g. $ ls $CSJDATATOP(USB) => 00README.txt DOC MORPH ... WAV fileList.csv
                ## Case merl :MERL setup. Neccesary directory is WAV and sdb
    HKUST1=
    HKUST2=
    LIBRISPEECH=
    JSUT=
    TIMIT=
    VOXFORGE=
    VIVOS=
    YESNO=
    HOW2_TEXT=
    HOW2_FEATS=

else
    AISHELL=downloads
    AN4=downloads
    WSJ0=
    WSJ1=
    WSJCAM0=
    REVERB=
    CHIME3=
    CHIME4=
    CSJDATATOP=
    CSJVER=dvd  ## Set your CSJ format (dvd or usb).
                ## Usage    :
                ## Case DVD : We assume CSJ DVDs are copied in this directory with the names dvd1, dvd2,...,dvd17.
                ##            Neccesary directory is dvd3 - dvd17.
                ##            e.g. $ ls $CSJDATATOP(DVD) => 00README.txt dvd1 dvd2 ... dvd17
                ##
                ## Case USB : Neccesary directory is MORPH/SDB and WAV
                ##            e.g. $ ls $CSJDATATOP(USB) => 00README.txt DOC MORPH ... WAV fileList.csv
                ## Case merl :MERL setup. Neccesary directory is WAV and sdb
    HKUST1=
    HKUST2=
    LIBRISPEECH=
    JSUT=downloads
    TIMIT=
    VOXFORGE=downloads
    VIVOS=downloads
    YESNO=downloads
    HOW2_TEXT=downloads/how2-300h-v1
    HOW2_FEATS=downloads/fbank_pitch_181516

fi
