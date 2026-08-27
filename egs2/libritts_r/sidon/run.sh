#!/usr/bin/env bash
set -euo pipefail

. ./path.sh
. ./cmd.sh
. ./db.sh

stage=1
stop_stage=7
ngpu=4
nj=64
python=python3
config=conf/train_sidon.yaml
expdir=exp/sidon_w2v_bert2_layer8
sidon_vocoder=
test_sets="test-clean test-other"

. utils/parse_options.sh

log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $*"; }

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: data preparation"
    local/data.sh
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: resample feature-predictor data to 16 kHz"
    for split in train dev; do
        mkdir -p data/${split}_fp_16k/wav
        ${python} local/resample_wav_scp.py \
            --input_scp data/${split}_fp/wav.scp \
            --output_scp data/${split}_fp_16k/wav.scp \
            --wav_dir data/${split}_fp_16k/wav \
            --target_sr 16000 --nj ${nj}
    done
    for test_set in ${test_sets}; do
        mkdir -p data/${test_set}_16k/wav
        ${python} local/resample_wav_scp.py \
            --input_scp data/${test_set}/wav.scp \
            --output_scp data/${test_set}_16k/wav.scp \
            --wav_dir data/${test_set}_16k/wav \
            --target_sr 16000 --nj ${nj}
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: generate RIR pool"
    ${python} local/prepare_rir_pool.py \
        --out_dir data/rir_pool --n_rirs 50000 --nj ${nj}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: collect feature-predictor statistics"
    ${python} -m espnet2.bin.enh_train_sidon \
        --config ${config} \
        --train_data_path_and_name_and_type data/train_fp_16k/wav.scp,speech_ref1,sound \
        --valid_data_path_and_name_and_type data/dev_fp_16k/wav.scp,speech_ref1,sound \
        --output_dir ${expdir} --collect_stats true --ngpu 0
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: train feature predictor"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        ${python} -m espnet2.bin.enh_train_sidon \
        --config ${config} \
        --train_data_path_and_name_and_type data/train_fp_16k/wav.scp,speech_ref1,sound \
        --valid_data_path_and_name_and_type data/dev_fp_16k/wav.scp,speech_ref1,sound \
        --train_shape_file ${expdir}/train/speech_ref1_shape \
        --valid_shape_file ${expdir}/valid/speech_ref1_shape \
        --output_dir ${expdir} --ngpu ${ngpu} \
        --multiprocessing_distributed true --unused_parameters true --resume true
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    [ -n "${sidon_vocoder}" ] || {
        log "Set --sidon_vocoder to official decoder_cuda.pt or decoder_cpu.pt"
        exit 1
    }
    for test_set in ${test_sets}; do
        log "Stage 6: inference (${test_set})"
        ${python} -m espnet2.bin.enh_inference_sidon \
            --train_config ${expdir}/config.yaml \
            --model_file ${expdir}/valid.loss.best.pth \
            --sidon_vocoder ${sidon_vocoder} \
            --wav_scp data/${test_set}_16k/wav.scp \
            --output_dir ${expdir}/inference_${test_set}
    done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    for test_set in ${test_sets}; do
        log "Stage 7: scoring (${test_set})"
        text_opt=()
        if [ -f "data/${test_set}/text" ]; then
            text_opt=(--text "data/${test_set}/text")
        fi
        ${python} local/score.py \
            --restored_dir ${expdir}/inference_${test_set}/wav \
            --ref_wav_scp data/${test_set}/wav.scp \
            --noisy_wav_scp data/${test_set}_16k/wav.scp \
            "${text_opt[@]}" \
            --output_dir ${expdir}/score_${test_set}
    done
fi
