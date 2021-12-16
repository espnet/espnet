#!/usr/bin/env bash

normdir=local/data_normalization
normfiles="shortened symbols url nlsyms"

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

# some patterns are treated beforehand (due to tokenization or ambiguity).
# it concerns apostrophe, simple brackets, squared bracked curly brackets,
# slash in urls, okay and hash terms, and dollars symbol.
cat ${intext} | \
    sed -r -e "s: &apos;:':g" \
        -e "s:&#91;:\[:g" \
        -e "s:&#93;:\]:g" \
        -e "s:\([^)]*\)::g" \
        -e "s:\[[^]]*\]::g" \
        -e "s:\{[^]]*\}::g" \
        -e "s|^[A-Za-z ]+:||g" \
        -e "s:(\.com|\.org) /:\1 slash:g" \
        -e "s:(^|[:punct:]| )(o|O)(\.)*(k|K)( |[:punct:]|$): okay :g" \
        -e "s:\\$ 10-plant:10 dollars plant:g" \
        -e "s:\\$ 3.50's: 3.50's dollars:g" \
        -e "s:(C|D|F) #:\1 sharp:g" \
        -e "s:#:number:g" \
        -e 's:(\$) ([0-9[:punct:]]*)($| ):\2 dollars \3:g' \
        > ${outtext}.tmp

_left_s="(^| )"
_right_s="( |$)"
_left_m="(^| |[[:punct:]])"
_right_m="([[:punct:]]| |$)"

# further normalization with simple left-right pattern and common terms
for file in $normfiles; do
    while read line || [ -n "$line" ]; do
        before=$(echo $line | cut -d ' ' -f1)
        after=$(echo $line | cut -d ' ' -f2-)

        if [ $file == "shortened" ]; then
            sed -i -r "s:${_left_m}${before}${_right_m}: ${after} :g" ${outtext}.tmp
        elif [ $file == "url" ] || [ $file == "symbols" ]; then
            sed -i -r "s:${before}: ${after} :g" ${outtext}.tmp
        elif [ $file == "nlsyms" ]; then
            sed -i -r "s:${_left_s}${before}${_right_m}: ${after} :g" ${outtext}.tmp
        else
            log "Error: unknown normalization files"
        fi
    done < "${normdir}/${file}"
done

# final punctuation and number normalization
cat ${outtext}.tmp | \
    sed -r -e "s:': APOSTROPHE :g" \
        -e "s:([0-9]) *(x|X) *([0-9]):\1 by \3:g" \
        -e "s:,([0-9]{3}):\1:g" \
        -e "s:\.(0)+${_right_s}: :g" \
        -e "s:([0-9])(\.)([0-9]):\1 point \3:g" \
        -e "s:[[:punct:]]: :g" \
        -e "s: APOSTROPHE :':g" \
        -e "s:hesmark:\[hes\]:g" \
        -e "s:^ +::g" \
        -e "s: +$::g" | \
    tr -s ' ' > ${outtext}.tmp2

basedir=$(dirname $intext)

# Remove empty lines and save line number to modify other needed files.
grep -n -e "^$" ${outtext}.tmp2 | cut -d':' -f1 | sed -r '/^ *$/d' \
                                                      > ${basedir}/utterances.deleted

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
