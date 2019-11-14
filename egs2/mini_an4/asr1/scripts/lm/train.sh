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
mkdir -p ${lmdatadir}

lmdatadir=${dir}/lm_train
lmdict=${lmdatadir}/vocab_${lm_vocabsize}.txt


if [ -n "${bpemodel}" ]; then
    mkdir -p ${lmdatadir}
    cp ${train_txt}> ${lmdatadir}/train.txt
    cp ${dev_txt} > ${lmdatadir}/valid.txt
    text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt

else
    if [ -n "${bpedict}" ]; then
        log "Error: --bpedict must be given"
        log "${help_message}"
        exti 1
    fi

    text2token.py -s 1 -n 1 ${train_txt} | cut -f 2- -d" " \
        | spm_encode --model=${bpemodel} --output_format=piece > ${lmdatadir}/train.txt

    text2token.py -s 1 -n 1 ${dev_txt}/text | cut -f 2- -d" " \
        | spm_encode --model=${bpemodel} --output_format=piece > ${lmdatadir}/valid.txt
fi


python -m espnet2.bin.train lm --print_config\
    --config="${train_config}" \
    --dict "${lmdict}"
    --train_data_conf=input.path="${lmdatadir}/train.txt" \
    --train_data_conf=input.type=text_int \
    --train_batch_files="[${lmdatadir}/train_text_shape]" \
    --eval_data_conf=input.path="${lmdatadir}/dev.txt" \
    --eval_data_conf=input.type=text_int \
    --eval_batch_files="[${devdir}/dev_text_shape]" \
    --print_config > "${expdir}/train.yaml"

train_config=${expdir}/train.yaml

${cmd} --gpu ${ngpu} ${dir}/train.log \
    python -m espnet2.lmbin.train_rnn \
    --config ${train_config} \
    --ngpu ${ngpu} \
    --output_dir "${expdir}/results"

log "Successfully finished. [elapsed=${SECONDS}s]"
