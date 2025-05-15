#!/usr/bin/env bash

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Do both Codec tokenization and SSL tokenization. Then splice the two kinds of
# discrete tokens together.

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

SECONDS=0

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=100
nj=4
fs=16000
file_name=
src_dir=
tgt_dir=

# codec settings
codec_choice=ESPnet
codec_checkpoint_path=null
codec_config_path=null
codec_hf_model_tag=null
codec_dump_audio=false
codec_batch_size=3

# ssl settings
ssl_choice=espnet_hubert
ssl_feature_type=wavlm_large
ssl_checkpoint_path=null
ssl_kmeans_path=null
ssl_nlayer=16
ssl_hf_model_tag=null
ssl_batch_bins=4800000

use_gpu=true
tolerance=1
python=python3

log "$0 $*"
. utils/parse_options.sh

. ./path.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Codec Tokenization ..."
    scripts/feats/codec_tokenization.sh \
      --src_dir ${src_dir} \
      --tgt_dir ${tgt_dir} \
      --file_name ${file_name} \
      --codec_fs ${fs} \
      --nj ${nj} \
      --batch_size ${codec_batch_size} \
      --dump_audio ${codec_dump_audio} \
      --codec_choice ${codec_choice} \
      --checkpoint_path ${codec_checkpoint_path} \
      --config_path ${codec_config_path} \
      --hf_model_tag ${codec_hf_model_tag}

    mv ${tgt_dir}/${file_name} ${tgt_dir}/codec_${file_name}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "SSL Tokenization ..."
    scripts/feats/ssl_tokenization.sh \
      --src_dir ${src_dir} \
      --tgt_dir ${tgt_dir} \
      --file_name ${file_name} \
      --fs ${fs} \
      --nj ${nj} \
      --batch_bins ${ssl_batch_bins} \
      --ssl_choice ${ssl_choice} \
      --ssl_feature_type ${ssl_feature_type} \
      --checkpoint_path ${ssl_checkpoint_path} \
      --kmeans_path ${ssl_kmeans_path} \
      --nlayer ${ssl_nlayer} \
      --hf_model_tag ${ssl_hf_model_tag}

    mv ${tgt_dir}/${file_name} ${tgt_dir}/ssl_${file_name}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Splice SSL and codec codes"

    file_name=${file_name%.scp}

    prefix=${tgt_dir}/data/${file_name}_codec_ssl_${codec_choice}.JOB
    wspecifier="ark,scp:${prefix}.ark,${prefix}.scp"

    ssl_vocab_size=$(cat ${tgt_dir}/token_lists/ssl_token_list | wc -l)
    codec_code_per_frame=$(cat ${tgt_dir}/token_lists/codec_code_per_frame)

    cat ${tgt_dir}/token_lists/{ssl,codec}_token_list > ${tgt_dir}/token_lists/codec_ssl_token_list

    ${decode_cmd} JOB=1:"${nj}" ${tgt_dir}/logdir/splice.JOB.log \
      ${python} pyscripts/feats/splice_scp.py \
        --ssl_scp ${tgt_dir}/data/${file_name}_ssl_${ssl_choice}.JOB.scp \
        --codec_scp ${tgt_dir}/data/${file_name}_codec_${codec_choice}.JOB.scp \
        --wspecifier ${wspecifier} \
        --tolerance ${tolerance} \
        --ssl_vocab_size ${ssl_vocab_size} \
        --codec_code_per_frame ${codec_code_per_frame}

    for n in `seq ${nj}`; do
        cat ${tgt_dir}/data/${file_name}_codec_ssl_${codec_choice}.${n}.scp
    done > ${tgt_dir}/${file_name}.scp
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
