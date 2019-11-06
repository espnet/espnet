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
$0 <train_set_name> <dev_set_name>

Options:
    --stage (int):
    --stop-stage (int):
    --bpemode (str):
    --nbpe (int):
    --bpdir (str):

EOF
)
SECONDS=0


stage=1
stop_stage=100

# bpemode (unigram or bpe)
bpemode=
nbpe=30
bpedir=

trans_type=char
# non-linguistic symbol list
nlsyms=
oov="<unk>"

log "$0 $*"
. ./utils/parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
    log "Invalid arguments"
    log "${help_message}"
fi

. ./path.sh

train_set=$1
train_dev=$2


[ -z "${bpedir}" ] && bpedir=data/local/bpe_${train_set}_${bpemode}${nbpe}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ -n "${bpemode}" ]; then
        dict=${bpedir}/units.txt
        bpemodel=${bpedir}/model
        mkdir -p ${bpedir}
        cut -f 2- -d" " ${train_set}/text > ${bpedir}/train.txt

        spm_train \
            --input=${bpedir}/train.txt \
            --vocab_size=${nbpe} \
            --model_type=${bpemode} \
            --model_prefix=${bpemodel} \
            --input_sentence_size=100000000

        echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        spm_encode --model=${bpemodel}.model --output_format=piece \
            < ${bpedir}/train.txt | \
            tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}

    else
        echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        text2token.py -s 1 -n 1 -l ${nlsyms} ${train_set}/text \
            | cut -f 2- -d" " | tr " " "\n" | sort -u \
            | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}

    fi

    # We prepares three type expressions for "text":
    #   text: The raw transcription
    #   token: Tokenized version of text
    #   token_int: The int-sequences converted from tokens.
    for dset in ${train_set} ${train_dev}; do

        if [ -n "${bpemode}" ]; then
            out=${dset}_bpe_${train_set}_${bpemode}${nbpe}
            utils/copy_data_dir.sh data/${dset} ${out}
            paste -d " " \
                <(awk '{print $1}' ${out}/text) \
                <(cut -f 2- -d" " ${out}/text |
                spm_encode --model=${bpemodel}.model --output_format=piece) \
                > ${out}/token

        else
            out=${dset}
            if [ -n "${nlsyms}" ]; then
                text2token.py -s 1 -n 1 -l ${nlsyms} data/${dset}/text \
                    --trans_type ${trans_type} > data/${dset}/token
            else
                text2token.py -s 1 -n 1 data/${dset}/text \
                    --trans_type ${trans_type} > data/${dset}/token
            fi
        fi

        <${out}/token \
            utils/sym2int.pl --map-oov ${oov} -f 2- ${dict} \
                > ${out}/token_int

        # Creating token_shape, which looks like...
        #   uttidA 20,32
        #   uttidB 12,32
        # where the first column indicates the number of tokens
        # and the secod is the vocabsize
        vocsize=$(tail -n 1 ${dict} | awk '{print $2}')
        # +2 comes from CTC blank and EOS
        odim="$((vocsize + 2))"
        <${out}/token_int \
            awk -v odim=${odim} '{print($1,NF-1 "," odim)}' > ${out}/token_shape
    done

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
