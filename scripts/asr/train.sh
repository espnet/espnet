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
$0 <train_set_dir> <dev_set_dir> <expdir>

Options:
    --task (str): Specify the task type.
    --preprocess-config (str): The configuration file for preprocessing for mini-batch. default=conf/preprocess.yaml
    --train-config (str):
EOF
)
SECONDS=0


cmd=utils/run.pl
ngpu=
preprocess_config=conf/preprocess.yaml

task=transformer

# mini-batch related
batch_type=const
batch_size=

train_args=
train_config=

log "$0 $*"

. ./utils/parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
    log "Invalid arguments"
    log "${help_message}"
fi

. ./path.sh

traindir=$1
devdir=$2
expdir=$3


for d in ${traindir} ${devdir}; do
    for f in wav.scp utt2num_samples token_int token_shape; do
        if [ ! -f ${d}/${f} ]; then
            log "Error: ${d}/${f} is not existing."
            exit 1
        fi
    done
done


if [ -n "${train_config}" ]; then
    log "Using ${train_config} for training configuration"
    cp ${train_config} ${expdir}/train.yaml
    train_config=${expdir}/train.yaml
else

    if [ "${task}" = transformer ]; then
        default_config=conf/train_transformer.yaml

    elif [ "${task}" = rnn ]; then
        default_config=conf/train_rnn.yaml

    elif [ "${task}" = rnnt ]; then
        default_config=conf/train_rnnt.yaml

    else
        log "Error: Not supported task: --task ${task}"
        exit 1
    fi

    train_config=${expdir}/train.yaml
    ./pyscripts/text/change_yaml.py ${default_config} ${train_args} -o ${train_config}

fi


# The configuration about mini-batch-IO for DNN training
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
    preprocess: {}
eval_data_config:
    data:
        input:
            path: ${devdir}/wav.scp
            type: sound
        output:
            path: ${devdir}/token_int
            type: text_int
    preprocess: {}

# For BatchSampler class
train_batch_config:
    type: "${batch_type}"
    batch_size: ${batch_size}
    shapes:
        - ${traindir}/utt2num_samples
        - ${traindir}/token_shape
eval_batch_config:
    type: "${batch_type}"
    batch_size: ${batch_size}
    shapes:
        - ${devdir}/utt2num_samples
        - ${devdir}/token_shape
EOF


if [ ! -z "${preprocess_config}" ]; then
    # Embed preprocess_config to train.yaml
    python3 << EOF
import yaml
with open('${train_config}', 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)
with open('${preprocess_config}', 'r') as f:
    preprocess_config = yaml.load(f, Loader=yaml.Loader)

# Embed preprocess_config and overwrite the config
config['train_data_config']['preprocess']['input'] = preprocess_config
config['eval_data_config']['preprocess']['input'] = preprocess_config
with open('${train_config}', 'w') as fout:
    yaml.dump(config, fout, Dumper=yaml.Dumper)
EOF
fi


log "Training started... log: ${expdir}/train.log"
${cmd} --gpu "${ngpu}" ${expdir}/train.log \
    python -m espnet2.bin.train "asr_${task}" \
        --ngpu "${ngpu}" \
        --config "${train_config}" \
        --output_dir ${expdir}/results \

log "Successfully finished. [elapsed=${SECONDS}s]"
