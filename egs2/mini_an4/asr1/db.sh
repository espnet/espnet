if [[ "$(hostname -d)" == clsp.jhu.edu ]]; then
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
    LIBRISPEECH=
    VOXFORGE=

elif [[ "$(hostname -d)" == cslab.kecl.ntt.co.jp ]]; then
    WSJ0=/data/rigel1/corpora/LDC93S6A
    WSJ1=/data/rigel1/corpora/LDC94S13A
    WSJCAM0=/data/rigel1/corpora/REVERB_DATA_OFFICIAL/wsjcam0
    REVERB=/data/rigel1/corpora/REVERB_DATA_OFFICIAL
    CHIME3=/data/rigel1/corpora/CHiME3
    CHIME4=/data/rigel1/corpora/CHiME4
    CSJDATATOP=/data/rigel1/corpora/CSJ
    CSJVER=dvd  ## Set your CSJ format (dvd or usb).
                ## Usage    :
                ## Case DVD : We assume CSJ DVDs are copied in this directory with the names dvd1, dvd2,...,dvd17.
                ##            Neccesary directory is dvd3 - dvd17.
                ##            e.g. $ ls $CSJDATATOP(DVD) => 00README.txt dvd1 dvd2 ... dvd17
                ##
                ## Case USB : Neccesary directory is MORPH/SDB and WAV
                ##            e.g. $ ls $CSJDATATOP(USB) => 00README.txt DOC MORPH ... WAV fileList.csv
                ## Case merl :MERL setup. Neccesary directory is WAV and sdb
    LIBRISPEECH=/data/rigel2/corpora/LibriSpeech
    VOXFORGE=

else
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
    LIBRISPEECH=
    VOXFORGE=downloads

fi
