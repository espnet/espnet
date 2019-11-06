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
$0 <train_set_dir> <dev_set_dir>
EOF
)
SECONDS=0


cmd=utils/run.pl
ngpu=

task=transformer

# mini-batch related
batch_type=const
batch_size=

train_args=
train_config=

log "$0 $*"

. ./utils/parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
    log "Invalid arguments"
    log "${help_message}"
fi

. ./path.sh


traindir=$1
devdir=$2


if [ "${task}" = transformer ]; then
    base_config=conf/train_transformer.yaml
    command_name=espnet2.asrbin.train_transformer

elif [ "${task}" = rnn ]; then
    log "Error: not yet"
    exit 1

else
    log "Error: Not supported task: --task ${task}"
    exit 1
fi

for dset in ${traindir} ${devdir}; do
    if [ -f ${dset}/feat_shape ]; then
        steps/feat2shape.sh --nj ${nj} --preprocess-config "${preprocess_config}" ${dset}/wav.scp ${dset}/feat_shape
    fi
done

if [ -n "${train_config}" ]; then
    train_config=$(change_yaml.py ${base_config} ${train_args} -o ${expdir})
    [ -f "${train_config}" ] && { log "Error: ${train_config} is not found, so maybe change_yaml.py was failed."; exit 1; }

    cat << EOF >> ${train_config}
# For Dataset class
train_data_config:
    data:
        input:
            path: ${traindir}/wav.scp
            type: sound
        output:
            path: ${traindir}/token_int
            type: text_int
    preprocess:
        input:
$(<${preprocess_config} awk '{ print "          " $1 }')
eval_data_config:
    data:
        input:
            path: ${devdir}/wav.scp
            type: sound
        output:
            path: ${devdir}/token_int
            type: text_int
    preprocess:
        input:
$(<${preprocess_config} awk '{ print "          " $1 }')

# For BatchSampler class
train_batch_config:
    type: "${batch_type}"
    batch_size: ${batch_size}
    shapes:
        - ${traindir}/feat_shape
        - ${traindir}/token_shape
eval_batch_config:
    type: "${batch_type}"
    batch_size: ${batch_size}
    shapes:
        - ${devdir}/feat_shape
        - ${devdir}/token_shape
EOF
fi

${cmd} --gpu "${ngpu}" ${expdir}/train.log \
    python -m ${command_name} \
        --ngpu "${ngpu}" \
        --config "${train_config}" \
        --outdir ${expdir}/results

log "Successfully finished. [elapsed=${SECONDS}s]"
