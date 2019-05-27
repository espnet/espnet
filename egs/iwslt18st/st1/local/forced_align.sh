#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

datadir=$1
train_data=$2

tmpdir=$(mktemp -d ${train_data}/../tmp-XXXXX)
trap 'cp ${tmpdir}/reclist local/reclist && rm -rf ${tmpdir}' EXIT

reclist=${tmpdir}/reclist
restart=false
last_line=""
if [ -f local/reclist ]; then
    restart=true
    cp local/reclist ${reclist}
    last_line=$(tail -1 ${reclist})
else
    touch ${reclist}
fi

cat ${train_data}/text.en | while read line; do
  utt_id=ted_`echo $line | cut -f 1 -d " " | cut -f 2- -d '_'`
  session_id=`echo $line | cut -f 1 -d " " | cut -f 2 -d "_" | sed 's/0*\([0-9]*[0-9]$\)/\1/g'`
  wav=${datadir}/train/iwslt-corpus/wav/ted_${session_id}.wav
  start_time=`echo $line | cut -f 1 -d " " | cut -f 3 -d "_" | awk '{ printf("%.2f", $1/1000); }'`
  end_time=`echo $line | cut -f 1 -d " " | cut -f 4 -d "_" | awk '{ printf("%.2f", $1/1000); }'`
  duration=`echo $line | awk '{
      segment=$1; split(segment,S,"[_]");
      spkid=S[1] "_" S[2]; startf=S[3]; endf=S[4];
      printf("%.2f", (endf-startf)/1000);
  }'`

  echo ${line}

  # skip until the last line of reclist
  if ${restart}; then
    if [ ${utt_id} = ${last_line} ]; then
        restart=false
    fi
    continue
  fi

  if [ $(echo "${duration} <= 30" | bc) == 1 ] && [ $(echo "${duration} >= 0.3" | bc) == 1 ]; then
      trans=${tmpdir}/${utt_id}.txt
      echo $line | cut -f 2- -d " " | detokenizer.perl -l en | local/remove_punctuation.pl > ${trans}

      cat ${trans}
      echo ${duration}

      # trim wav files
      trim_wav=${tmpdir}/${utt_id}.wav
      sox ${wav} ${trim_wav} trim ${start_time} ${duration} || exit 1;

      # forced-align
      log=${tmpdir}/${utt_id}.log
      timeout 30 python3 /home/inaguma/tool/gentle/align.py ${trim_wav} ${trans} > ${log}

      STATUS=$?
      if [ "$STATUS" -ne 124 ]; then
        # remove all utterances in which at least one word in the gold transcription was not aligned
        if [ $(cat ${log} | grep -ic 'not-found-in-audio') = 0 ]; then
          echo ${utt_id} >> ${reclist}
        fi
      fi

      rm ${log} ${trim_wav} ${trans}
  fi
done
