# Set the path of your corpus
# "downloads" means the corpus can be downloaded by the recipe automatically

AISHELL=downloads
AN4=downloads
DIRHA_ENGLISH_PHDEV=
DIRHA_WSJ=
DIRHA_WSJ_PROCESSED="${PWD}/data/local/dirha_wsj_processed"  # Output file path
DNS=
WSJ0=
WSJ1=
WSJCAM0=
REVERB=
REVERB_OUT="${PWD}/REVERB"  # Output file path
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
CSMSC=downloads
HKUST1=
HKUST2=
LABOROTV=
TEDXJP=
LIBRISPEECH=
MINI_LIBRISPEECH=downloads
LIBRITTS=
LJSPEECH=downloads
NSC=
JSSS=downloads
JSUT=downloads
JVS=downloads
SPGISPEECH=
SWBD=
TIMIT=$(realpath ../../../../TIMIT)
VOXFORGE=downloads
AMI=
COMMONVOICE=downloads
BABEL_101=
BABEL_102=
BABEL_103=
BABEL_104=
BABEL_105=
BABEL_106=
BABEL_107=
BABEL_201=
BABEL_202=
BABEL_203=
BABEL_204=
BABEL_205=
BABEL_206=
BABEL_207=
BABEL_301=
BABEL_302=
BABEL_303=
BABEL_304=
BABEL_305=
BABEL_306=
BABEL_307=
BABEL_401=
BABEL_402=
BABEL_403=
BABEL_404=
PUEBLA_NAHUATL=
TEDLIUM3=downloads
VCTK=downloads
VIVOS=downloads
YESNO=downloads
YOLOXOCHITL_MIXTEC=downloads
HOW2_TEXT=downloads/how2-300h-v1
HOW2_FEATS=downloads/fbank_pitch_181516
ZEROTH_KOREAN=downloads
RU_OPEN_STT=downloads
GIGASPEECH=
NOISY_SPEECH=
NOISY_REVERBERANT_SPEECH=

# For only JHU environment
if [[ "$(hostname -d)" == clsp.jhu.edu ]]; then
    AISHELL=
    AN4=
    DIRHA_ENGLISH_PHDEV=
    DIRHA_WSJ=
    DIRHA_WSJ_PROCESSED="${PWD}/data/local/dirha_wsj_processed"  # Output file path
    DNS=
    WSJ0=
    WSJ1=
    WSJCAM0=/export/corpora3/LDC/LDC95S24/wsjcam0
    REVERB=/export/corpora5/REVERB_2014/REVERB
    REVERB_OUT="${PWD}/REVERB"  # Output file path
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
    CSMSC=downloads
    HKUST1=
    HKUST2=
    LABOROTV=
    TEDXJP=
    LIBRISPEECH=
    MINI_LIBRISPEECH=downloads
    LIBRITTS=
    LJSPEECH=downloads
    JSSS=downloads
    JSUT=downloads
    JVS=downloads
    TIMIT=
    VOXFORGE=
    AMI=/export/corpora4/ami/amicorpus
    COMMONVOICE=downloads
    BABEL_101=/export/babel/data/101-cantonese
    BABEL_102=/export/babel/data/102-assamese
    BABEL_103=/export/babel/data/103-bengali
    BABEL_104=/export/babel/data/104-pashto
    BABEL_105=/export/babel/data/105-turkish
    BABEL_106=/export/babel/data/106-tagalog
    BABEL_107=/export/babel/data/107-vietnamese
    BABEL_201=/export/babel/data/201-haitian
    BABEL_202=/export/babel/data/202-swahili/IARPA-babel202b-v1.0d-build/BABEL_OP2_202
    BABEL_203=/export/babel/data/203-lao
    BABEL_204=/export/babel/data/204-tamil
    BABEL_205=/export/babel/data/205-kurmanji/IARPA-babel205b-v1.0a-build/BABEL_OP2_205
    BABEL_206=/export/babel/data/206-zulu
    BABEL_207=/export/babel/data/207-tokpisin/IARPA-babel207b-v1.0e-build/BABEL_OP2_207
    BABEL_301=/export/babel/data/301-cebuano/IARPA-babel301b-v2.0b-build/BABEL_OP2_301
    BABEL_302=/export/babel/data/302-kazakh/IARPA-babel302b-v1.0a-build/BABEL_OP2_302
    BABEL_303=/export/babel/data/303-telugu/IARPA-babel303b-v1.0a-build/BABEL_OP2_303
    BABEL_304=/export/babel/data/304-lithuanian/IARPA-babel304b-v1.0b-build/BABEL_OP2_304
    BABEL_305=/export/babel/data/305-guarani/IARPA-babel305b-v1.0b-build/BABEL_OP3_305
    BABEL_306=/export/babel/data/306-igbo/IARPA-babel306b-v2.0c-build/BABEL_OP3_306
    BABEL_307=/export/babel/data/307-amharic/IARPA-babel307b-v1.0b-build/BABEL_OP3_307
    BABEL_401=/export/babel/data/401-mongolian/IARPA-babel401b-v2.0b-build/BABEL_OP3_401
    BABEL_402=/export/babel/data/402-javanese/IARPA-babel402b-v1.0b-build/BABEL_OP3_402
    BABEL_403=/export/babel/data/403-dholuo/IARPA-babel403b-v1.0b-build/BABEL_OP3_403
    BABEL_404=/export/corpora/LDC/LDC2016S12/IARPA_BABEL_OP3_404
    PUEBLA_NAHUATL=
    TEDLIUM3=downloads
    VCTK=downloads
    VIVOS=
    YESNO=
    YOLOXOCHITL_MIXTEC=downloads
    HOW2_TEXT=
    HOW2_FEATS=
    ZEROTH_KOREAN=downloads

fi
