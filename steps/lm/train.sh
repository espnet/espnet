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
SECONDS=0

cmd=utils/run.pl
ngpu=0

lm_config=conf/lm.yaml

# rnnlm related
use_wordlm=false    # false means to train/use a character LM
lm_vocabsize=100    # effective only for word LMs

bpemodel=
bpedict=


# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;
. ./path.sh

train_txt=$1
dev_txt=$1
test_txt=$1

dir=$1

mkdir -p ${dir}

if [ ${use_wordlm} = true ]; then
    lmdatadir=${dir}/wordlm_train
    lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
    mkdir -p ${lmdatadir}
    cut -f 2- -d" " ${train_txt}> ${lmdatadir}/train.txt
    cut -f 2- -d" " ${dev_txt}/text > ${lmdatadir}/valid.txt
    cut -f 2- -d" " ${test_txt}/text > ${lmdatadir}/test.txt
    text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
else
    if [ -n "${bpedict}" ]; then
        log "Error: --bpedict must be given"
        log "${help_message}"
        exti 1
    fi

    lmdatadir=${dir}/lm_train
    mkdir -p ${lmdatadir}

    text2token.py -s 1 -n 1 ${train_txt} \
        | cut -f 2- -d" " \
        | spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt

    text2token.py -s 1 -n 1 ${dev_txt}/text \
        | cut -f 2- -d" " \
        | spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/valid.txt

    text2token.py -s 1 -n 1 ${test_txt}/text \
        | cut -f 2- -d" " \
        | spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/test.txt
fi

log "Not created yet"
exit 1


${cmd} --gpu ${ngpu} ${dir}/train.log \
    python -m espnet2.lmbin.train_rnn \
    --config ${lm_config} \
    --ngpu ${ngpu} \
    --verbose 1 \
    --outdir ${dir} \
    --train-label ${lmdatadir}/train.txt \
    --valid-label ${lmdatadir}/valid.txt \
    --test-label ${lmdatadir}/test.txt \
    --resume ${lm_resume} \
    --dict ${lmdict} \
    --dump-hdf5-path ${lmdatadir}

log "Successfully finished. [elapsed=${SECONDS}s]"
