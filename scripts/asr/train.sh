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
        if [ ! -f "${d}/${f}" ]; then
            log "Error: ${d}/${f} is not existing."
            exit 1
        fi
    done
done


if [ "${task}" != transformer ] && \
   [ "${task}" != rnn ] && \
   [ "${task}" != rnnt ]; then
    log "Error: Not supported task: --task ${task}"
    exit 1
fi



if [ -n "${train_config}" ]; then
    log "Copying ${train_config} to ${expdir}/train.yaml"
    cp ${train_config} "${expdir}/train.yaml"
else
    python -m espnet2.bin.train "asr_${task}" --show_config > "${expdir}/train.yaml"
fi
train_config=${expdir}/train.yaml


# The configuration about mini-batch-IO for DNN training
pyscripts/text/change_yaml.py \
    "${train_config}" \
    -a train_data_conf.input.path="${traindir}/wav.scp" \
    -a train_data_conf.input.type=sound \
    -a train_data_conf.output.path="${traindir}/token_int" \
    -a train_data_conf.output.type=text_int \
    -a train_batch_files="[${traindir}/utt2num_samples, ${traindir}/token_shape]" \
    -a eval_data_conf.input.path="${devdir}/wav.scp" \
    -a eval_data_conf.input.type=sound \
    -a eval_data_conf.output.path="${devdir}/token_int" \
    -a eval_data_conf.output.type=text_int \
    -a eval_batch_files="[${devdir}/utt2num_samples, ${devdir}/token_shape]" \
    -o "${train_config}"  # Overwrite


if [ -n "${preprocess_config}" ]; then
    log "Embeding ${preprocess_config} in ${train_config}"
    python3 << EOF
import yaml
from copy import deepcopy
with open('${train_config}', 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)
with open('${preprocess_config}', 'r') as f:
    preprocess_config = yaml.load(f, Loader=yaml.Loader)

# Embed preprocess_config and overwrite the config
config['train_preprocess']['input'] = deep_copy(preprocess_config)
config['eval_preprocess']['input'] = deep_copy(preprocess_config)
with open('${train_config}', 'w') as fout:
    yaml.dump(config, fout, Dumper=yaml.Dumper, indent=4, sort_keys=False)
EOF
fi


log "Training started... log: ${expdir}/train.log"
${cmd} --gpu "${ngpu}" "${expdir}/train.log" \
    python -m espnet2.bin.train "asr_${task}" \
        --ngpu "${ngpu}" \
        --config "${train_config}" \
        --output_dir "${expdir}/results"


log "Successfully finished. [elapsed=${SECONDS}s]"
