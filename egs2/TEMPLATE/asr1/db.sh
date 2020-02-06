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
    EVAL2000="/export/corpora2/LDC/LDC2002S09/hub5e_00 /export/corpora2/LDC/LDC2002T43"
    FISHER="/export/corpora3/LDC/LDC2004T19 /export/corpora3/LDC/LDC2005T19 /export/corpora3/LDC/LDC2004S13 /export/corpora3/LDC/LDC2005S13"
    LIBRISPEECH=
    RT03=/export/corpora/LDC/LDC2007S10
    SWBD=/export/corpora3/LDC/LDC97S62
    VOXFORGE=
    VIVOS=
    YESNO=

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
    EVAL2000=
    FISHER=
    LIBRISPEECH=
    RT03=
    SWBD=
    VOXFORGE=downloads
    VIVOS=downloads
    YESNO=downloads

fi
