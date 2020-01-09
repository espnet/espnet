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
    YESNO=
    VIVOS=

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
    YESNO=
    VIVOS=

fi
