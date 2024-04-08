#!/usr/bin/env bash
score_lang_id=true
dump_dir=dump
. utils/parse_options.sh

if [ $# -gt 1 ]; then
    echo "Usage: $0 --score_lang_id ture [exp]" 1>&2
    echo ""
    echo "Language Identification Scoring."
    echo 'The default of <exp> is "exp/".'
    exit 1
fi

[ -f ./path.sh ] && . ./path.sh
set -euo pipefail
if [ $# -eq 1 ]; then
    exp=$1
else
    exp=exp
fi

if [ "${score_lang_id}" = false ]; then
    echo "Training without language id, skip language identification scoring"
    exit 1
fi

for infer in $(ls -d ${exp}/decode_*); do
    if [ ! -d $infer ]; then
        continue
    fi
    echo "Scoring Lang id for ${infer}"
    for dset in $(ls -d ${infer}/*/); do
        echo ${dset}
        _data="${dump_dir}/raw/$(basename ${dset})"
        _dir="${dset}"

        _scoredir="${_dir}/score_lang_id"
        mkdir -p "${_scoredir}"

        paste \
            <(<"${_data}/text" \
            awk '{if (match($2, "\\[[a-zA-Z0-9_\\-]+\\]")) {
                lang_id = substr($2, RSTART, RLENGTH);
                printf("%s %s\n", $1, lang_id);  }} ') \
            <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
            >${_scoredir}/"ref.trn"

        if [ ! -f ${_dir}/text ]; then
            echo "no inference result for ${dset}. Skip it"
            continue
        fi

        paste \
            <(<"${_dir}/text" \
            awk '{if (match($2, "\\[[a-zA-Z0-9_\\-]+\\]")) {
                lang_id = substr($2, RSTART, RLENGTH);
                printf("%s %s\n", $1, lang_id);  }} ') \
            <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
            >${_scoredir}/"hyp.trn"

        score_lang_id.py --ref ${_scoredir}/ref.trn --hyp ${_scoredir}/hyp.trn \
                         --out ${_scoredir}/result.txt

        echo "Saved results at ${_scoredir}/result.txt"
        cat ${_scoredir}/result.txt
        echo ""
    done
done
