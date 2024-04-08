#!/usr/bin/env bash

for x in 101-cantonese 102-assamese 103-bengali 104-pashto 105-turkish 106-tagalog 107-vietnamese 201-haitian 202-swahili 203-lao 204-tamil 205-kurmanji 206-zulu 207-tokpisin 404-georgian; do
    langid=`echo $x | cut -f 1 -d"-"`
    lang=`echo $x | cut -f 2 -d"-"`
    echo $langid
    echo $lang
    ./setup_experiment.sh asr1_${lang}
    pushd ../asr1_${lang}
    ./run.sh --langs $langid --recog $langid --ngpu 1 &
    sleep 20m # to avoid too many disk access happened at the same time
    popd
done
