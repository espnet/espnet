#!/bin/bash
# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
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
Usage: $0
EOF

)
SECONDS=0

# general configuration
backend=pytorch
stage=-1
stop_stage=100
ngpu=0       # number of gpus ("0" uses cpu, otherwise use gpu)
nj=2         # numebr of parallel jobs
dumpdir=dump # directory to dump full features
verbose=1    # verbose option (if set > 0, get more log)
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=16000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# config files
train_config=conf/train_pytorch_mini_tacotron2.yaml
decode_config=conf/mini_decode.yaml

# decoding related
griffin_lim_iters=4  # the number of iterations of Griffin-Lim

# data
datadir=downloads
an4_root=${datadir}/an4

# exp tag
tag="" # tag for managing experiments.


decode_nj=50
exp=exp
preprocess_config=
feats_type=fbank
gpu_decode=false

# [Task depented] Set the datadir name created by local/data.sh
train_set=
dev_set=
eval_sets=

. utils/parse_options.sh

. ./path.sh
. ./cmd.sh


if [ "${feats_type}" = fbank ]; then
    data_feats=data_fbank
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi
data_tts="data_tts_${feats_type}"


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparation for data/${train_set}, data/${dev_set}, etc."
    local/data.sh
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Feature Generation"

    # TODO(kamo): Change kaldi-ark to npy or HDF5?

    # Generate the fbank features; by default 80-dimensional fbanks on each frame
    for dset in "${train_set}" "${dev_set}" ${eval_sets}; do
        utils/copy_data_dir.sh data/"${dset}" data_fbank/"${dset}"
        scritps/feats/make_fbank.sh --cmd "${train_cmd}" --nj "${nj}" \
            --fs "${fs}" \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft "${n_fft}" \
            --n_shift "${n_shift}" \
            --win_length "${win_length}" \
            --n_mels "${n_mels}"

        echo "fbank" > "data_fbank/${dset}/feats_type"
    done

    # compute statistics for global mean-variance normalization
    pyscripts/feats/compute-cmvn-stats.py \
        scp:"cat data/${train_set}/feats.scp data/${dev_set}/feats.scp |" \
        "data/${train_set}/cmvn.ark"

fi


dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    scripts/text/text2token.sh -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

fi


tts_exp=${exp}/train
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    _train_dir="${data_tts}/${train_set}"
    _dev_dir="${data_tts}/${dev_set}"
    log "stage 7: ASR Training: train_set=${_train_dir}, dev_set=${_dev_dir}"

    _opts=
    if [ -n "${preprocess_config}" ]; then
        # syntax: --train_preprosess {key}={yaml file or yaml string}
        _opts+="--train_preprosess input=${preprocess_config} "
        _opts+="--eval_preprosess input=${preprocess_config} "
    fi
    if [ -n "${config}" ]; then
        # To generate the config file: e.g.
        #   % python -m espnet2.bin.asr_train --print_config --optimizer adam --encoder_decoder transformer
        _opts+="--config ${config} "
    fi

    # FIXME(kamo): max_length is confusing name. How about fold_length?

    log "TTS training started... log: '${tts_exp}/train.log'"
    ${cuda_cmd} --gpu "${ngpu}" "${tts_exp}"/train.log \
        python3 -m espnet2.bin.asr_train \
            --ngpu "${ngpu}" \
            --token_list "${_train_dir}/tokens.txt" \
            --train_data_path_and_name_and_type "${_train_dir}/token_int,input,text_int" \
            --train_data_path_and_name_and_type "${_train_dir}/feats.scp,output,kaldi_ark" \
            --eval_data_path_and_name_and_type "${_dev_dir}/token_int,input,text_int" \
            --eval_data_path_and_name_and_type "${_dev_dir}/feats.scp,output,kaldi_ark" \
            --train_shape_file "${_train_dir}/token_shape" \
            --train_shape_file "${_train_dir}/feats_shape" \
            --eval_shape_file "${_dev_dir}/token_shape" \
            --eval_shape_file "${_dev_dir}/feats_shape" \
            --resume_epoch latest \
            --odim "$(<${_train_dir}/feats_shape head -n1 | cut -d ' ' -f 2 | cut -d',' -f 2)" \
            --max_length 150 \
            --max_length 800 \
            --output_dir "${expdir}" \
            ${_opts}

fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    log "stage 8: Decoding"

    if ${gpu_decode}; then
        _cmd=${cuda_cmd}
        _ngpu=1
    else
        _cmd=${decode_cmd}
        _ngpu=0
    fi

    _opts=
    if [ -n "${asr_preprocess_config}" ]; then
        _opts+="--preprosess input=${asr_preprocess_config} "
    fi
    if [ -n "${decode_config}" ]; then
        _opts+="--config ${decode_config} "
    fi

    for dset in ${eval_sets}; do
        _data="${data_asr}/${dset}"
        _dir="${asr_exp}/decode_${dset}${decode_tag}"
        _logdir="${_dir}/logdir"
        mkdir -p "${_logdir}"

        # 1. Split the key file
        key_file=${_data}/wav.scp
        split_scps=""
        _nj=$((${decode_nj}<$(<${key_file} wc -l)?${decode_nj}:$(<${key_file} wc -l)))
        for n in $(seq ${_nj}); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        utils/split_scp.pl "${key_file}" ${split_scps}

        _feats_type="$(<${_data}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            _type=sound
        else
            _scp=feats.scp
            _type=kaldi_ark
        fi


        # 2. Submit decoding jobs
        log "Decoding started... log: '${_logdir}/asr_recog.*.log'"
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/asr_recog.JOB.log \
            python3 -m espnet2.bin.asr_recog \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${_data}/${_scp},input,${_type}" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --asr_train_config "${asr_exp}"/config.yaml \
                --asr_model_file "${asr_exp}"/"${decode_asr_model}" \
                --lm_train_config "${lm_exp}"/config.yaml \
                --lm_file "${lm_exp}"/"${decode_lm}" \
                --output_dir "${_logdir}"/output.JOB \
                ${_opts}

        # 3. Concatenates the output files from each jobs
        for f in token token_int score; do
            for i in $(seq "${_nj}"); do
                cat "${_logdir}/output.${i}/1best_recog/${f}"
            done | LC_ALL=C sort -k1 >"${_dir}/${f}"
        done

        # 4. Convert token to text
        _token_type="$(<${_data}/token_type)"

        if [ "${_token_type}" = bpe ]; then
            paste <(<${_dir}/token cut -f 1 -d" ") \
                <(<${_dir}/token cut -f 2- -d" " \
                  | spm_decode --model=${bpemodel} --input_format=piece \
                  | sed -e "s/▁/ /g") \
                >  ${_dir}/text

        elif [ "${_token_type}" = char ]; then
            paste <(<${_dir}/token cut -f 1 -d" ") \
                <(<${_dir}/token cut -f 2- -d" "  \
                  | sed -e 's/ //g' \
                  | sed -e 's/<space>/ /g' \
                  ) \
                >  ${_dir}/text
        else
            log "Error: not supported --token_type '${_token_type}'"
            exit 2
        fi

    done
fi



if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Synthesis"
    pids=() # initialize pids
    for sets in ${train_dev} ${eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${sets} ] && mkdir -p ${outdir}_denorm/${sets}
        apply-cmvn --norm-vars=true --reverse=true data/${train_set}/cmvn.ark \
            scp:${outdir}/${sets}/feats.scp \
            ark,scp:${outdir}_denorm/${sets}/feats.ark,${outdir}_denorm/${sets}/feats.scp
        convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            --iters ${griffin_lim_iters} \
            ${outdir}_denorm/${sets} \
            ${outdir}_denorm/${sets}/log \
            ${outdir}_denorm/${sets}/wav
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi
