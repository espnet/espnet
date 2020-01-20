#!/bin/bash

normdir=local/data_normalization
# keep the file order
normfiles="shortened symbols url n_special dates n_ordinal nlsyms n_cardinal"

_left_s="(^| )"
_right_s="( |$)"
_left_m="(^| |[[:punct:]])"
_right_m="([[:punct:]]| |$)"

log() {
    local fname=${BASH_SOURCE[1]##*/}
        echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $@"
}
SECONDS=0

if [ $# -ne 4 ]; then
    log "Usage: $0 <in-text> <in-id-spk> <out-text> <out-id-spk>"
    exit 2
fi

intext=$1
idspk=$2
outtext=$3
outidspk=$4

[ ! -f ${intext} ] && log "${intext} was not found." && exit 2

# ambiguous regex in simple left-right context are treated beforehand.
# it concerns square brackets, brckets, curly brackets and dollars symbol.
cat ${intext} | \
    sed "s:([^)]*)::g" | sed -r "s:\[[^]]*\]::g" | \
    sed -r "s:\{[^]]*\}::g" | sed -r "s|^[A-Z ]+:||g" | \
    sed -r "s:(o|O)\.(k|K)\.: okay :g" | \
    sed -r 's:(\$)([0-9]*)([[:punct:]]$|$| ):\2 dollars \3:g' \
        > ${outtext}.tmp

# text normalization
for file in $normfiles; do
    while read line || [ -n "$line" ]; do
        before=$(echo $line | cut -d ' ' -f1)
        after=$(echo $line | cut -d ' ' -f2-)

        if [ $file == "shortened" ] || [ $file == "n_special" ] || [ $file == "n_ordinal" ]; then
            sed -i -r "s:${_left_m}${before}${_right_m}: ${after} :g" ${outtext}.tmp
        elif [ $file == "url" ] || [ $file == "symbols" ]; then
            sed -i -r "s:${before}: ${after} :g" ${outtext}.tmp
        elif [ $file == "n_cardinal" ]; then
            sed -i -r "s:${left_s}${before}${_right_s}: ${after} :g" ${outtext}.tmp
        elif [ $file == "nlsyms" ]; then
            sed -i -r "s:${_left_s}${before}${_right_m}: ${after} :g" ${outtext}.tmp
        elif [ $file == "dates" ]; then
            sed -i -r "s: ${before}${_right_m}: ${after} :g" ${outtext}.tmp
        else
            log "Error: $file is undefined for normalization."
        fi
    done < "${normdir}/${file}"
done

# punctuation normalization
cat ${outtext}.tmp | \
    sed -r -e "s:': APOSTROPHE :g" \
        -e "s:[[:punct:]]: :g" \
        -e "s: APOSTROPHE :':g" \
        -e "s:hesmark:[hes]:g" \
        -e "s:^ +::g" \
        -e "s: +$::g" | \
    tr -s ' ' > ${outtext}.tmp2

basedir=$(dirname $intext)

grep -n -e "^$" -e "ü" -e "è" -e "ñ" -e "é" -e "[0-9]" ${outtext}.tmp2 | \
    cut -d':' -f1 | sed -r '/^ *$/d' > ${basedir}/utterances.deleted

if [ -s ${basedir}/utterances.deleted ]; then
    awk 'NR==FNR{l[$0];next;} !(FNR in l)' ${basedir}/utterances.deleted ${outtext}.tmp2 \
        > ${outtext}

    awk 'NR==FNR{l[$0];next;} !(FNR in l)' ${basedir}/utterances.deleted ${idspk} \
        > ${outidspk}
else
    cp ${outtext}.tmp2 ${outtext}
    cp ${idspk} ${outidspk}
fi

rm ${outtext}.tmp*
