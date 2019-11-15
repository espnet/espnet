#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $@"
}
help_message=$(cat << EOF
$0 <text> <dict> <output-dir>

Generate token, token_int and token_shape from text

Options:
    --mode (str): The tokenize level. Select either one of "bpe", "char" or "word".
    --bpemodel (str): Give bpemodel with --mode bpe.
    --nlsyms: non-linguistic symbol list
    --oov (str): default is "<unk>"

EOF
)
SECONDS=0

mode=bpe
bpemodel=
nlsyms=
oov="<unk>"


log "$0 $*"
. ./utils/parse_options.sh || exit 1;


if [ $# -ne 3 ]; then
    log "Invalid arguments"
    log "${help_message}"
fi
. ./path.sh

text=$1
dict=$2
dir=$3

for f in "${text}" "${dict}"; do
    if [ ! -f "${f}" ]; then
        log "Error: No such file ${f}"
        log "${help_message}"
        exit 1
    fi
done


mkdir -p "${dir}"


# 1. Prepare token
if [ "${mode}" = bpe ]; then
    if [ ! -n "${bpemodel}" ]; then
        log "Error: --bpemodel is required for bpe mode."
        log "${help_message}"
        exit 1
    fi

    paste -d " " \
        <(awk '{print $1}' "${text}") \
        <(cut -f 2- -d" " "${text}" | spm_encode --model="${bpemodel}" --output_format=piece) \
            > ${dir}/token


elif [ "${mode}" = char ]; then
    trans_type=char
    # non-linguistic symbol list

    # token:
    # uttidA h e l l o
    if [ -n "${nlsyms}" ]; then
        pyscripts/text/text2token.py -s 1 -n 1 -l "${nlsyms}"" ${text}" \
            --trans_type "${trans_type}" > "${dir}/token"
    else
        pyscripts/text/text2token.py -s 1 -n 1 ${dset}/text \
            --trans_type "${trans_type}" > "${ddir}/token"
    fi

elif [ "${mode}" = word ]; then
    log "not yet"
    exit 1

else
    log "Error: not supported --mode ${mode}"
    exit 1
fi


# 3. Create "token_int"
<${dir}/token utils/sym2int.pl --map-oov "${oov}" -f 2- "${dict}" > "${dir}/token_int"


# 4. Create "token_shape", which looks like...
#   uttidA 20,32
#   uttidB 12,32
# where the first column indicates the number of tokens
# and the secod is the vocabsize
vocsize=$(tail -n 1 "${dict}" | awk '{print $2}')
# +2 comes from CTC blank and SOS/EOS
odim="$((vocsize + 2))"
<${dir}/token_int awk -v odim="${odim}" '{print($1,NF-1 "," odim)}' > ${dir}/token_shape


# 5. Copy dict: dict is a list of tokens
cp "${dict}" "${dir}/tokens.txt"


log "Successfully finished. [elapsed=${SECONDS}s]"
