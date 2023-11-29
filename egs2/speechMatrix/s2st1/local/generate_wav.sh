#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=0
stop_stage=100
gpu_inference=true
nj=2

vocoder_dir=exp/fairseq_hifigan_vocoder
vocoder_ckpt=${vocoder_dir}/vocoder.ckpt
vocoder_config=${vocoder_dir}/config.json

test_sets="dev_es test_es"
decode_dir=exp/s2st_train_s2st_discrete_unit_raw_fbank_es_en/decode_s2st_s2st_model_train.loss.best

. path.sh
. cmd.sh
. utils/parse_options.sh

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${vocoder_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Downloading pre-trained vocoder"

  if [ ! -f ${vocoder_ckpt} ]; then
    wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000 -O ${vocoder_ckpt}
    wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json -O ${vocoder_config}
  fi
fi

# Convert .npy files into text that contains a sequence of integers per line
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  log "Preparing discrete unit inputs"

  for dset in ${test_sets}; do
    out_dir=${decode_dir}/${dset}/discrete_units
    mkdir -p ${out_dir}

    python local/process_decode_output.py --feats_scp "${decode_dir}/${dset}/norm/feats.scp" --out_dir "${out_dir}"
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  log "Generating waveform using the pre-trained vocoder"

  for dset in ${test_sets}; do
    out_dir=${decode_dir}/${dset}/wav
    log_dir=${out_dir}/log
    mkdir -p ${log_dir}

    # split
    splits=""
    for n in $(seq "${nj}"); do
      splits+=" ${log_dir}/utt2units.${n}"
    done
    utils/split_scp.pl "${decode_dir}/${dset}/discrete_units/utt2units" ${splits}

    opts=""
    ngpu=0
    cmd="${decode_cmd}"
    if ${gpu_inference}; then
      cmd="${cuda_cmd}"
      ngpu=1
    else
      opts+="--cpu"
    fi

    log "Generating wavs... log: '${log_dir}/generate_wave_from_code.*.log'"
    ${cmd} --gpu "${ngpu}" JOB=1:"${nj}" "${log_dir}"/generate_wave_from_code.JOB.log \
      python local/generate_waveform_from_code.py \
      --utt2units "${log_dir}"/utt2units.JOB \
      --vocoder ${vocoder_ckpt} --vocoder_cfg ${vocoder_config} \
      --out_dir "${out_dir}"/output.JOB \
      --dur_prediction ${opts}

    cat "${out_dir}"/output.*/wav.scp >"${out_dir}"/wav.scp

    log "Output saved in ${out_dir}/wav.scp"
  done
fi
