#!/usr/bin/env bash

# Copyright 2020 Electronics and Telecommunications Research Institute (Jeong-Uk, Bang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. utils/parse_options.sh || exit 1;

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <db-dir> <text-dir>"
    echo "e.g.: $0 /mls/jubang/databases/KsponSpeech data/local/KsponSpeech"
    exit 1
fi

db=$1
text=$2

tgt_case="df"
# fl: fluent transcription
# df: disfluent transcription
# dt: disfluent transcription with tag symbols ('/' or '+')

[ ! -d ${db} ] && echo "$0: no such directory ${db}" && exit 1;
[ -f ${text}/.done ] && echo "$0: the KsponSpeech transcription exists ==> Skip" && exit 0;

mkdir -p ${text} ${text}/logs || exit 1;

# 1) get original transcription
for x in train dev eval_clean eval_other; do
    [ ! -f ${db}/scripts/${x}.trn ] && echo "$0: no such transcription scripts/${x}.trn" && exit 1;
    mkdir -p ${text}/${x} && cp -a ${db}/scripts/${x}.trn ${text}/${x}/text.raw
done

# 2) get transcription files
echo "$0: get transcription files for KsponSpeech"
for task in train dev eval_clean eval_other; do
    python local/get_transcriptions.py --verbose 1 --clear \
        --type $tgt_case --log-dir ${text}/logs/get_transcription --unk-sym '[unk]' \
        --raw-trans ${text}/${task}/text.raw --out-fn ${text}/${task}/text.trn
done

echo "$0: successfully prepared transcription files for KsponSpeech dataset"
touch ${text}/.done && exit 0;
