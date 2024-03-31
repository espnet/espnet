#!/usr/bin/env bash

set -euo pipefail

source tools/activate_python.sh
PYTHONPATH="${PYTHONPATH:-}:$(pwd)/tools/s3prl"
export PYTHONPATH
python="python -m coverage run --append"
cwd=$(pwd)

gen_dummy_coverage(){
    # To avoid a problem when parallel running for `coverage run`.
    # Please put this command after cd ./egs2/foo/bar
    touch empty.py; ${python} empty.py
}

#### Make sure chainer-independent ####
python3 -m pip uninstall -y chainer

# Download mini_an4 as test data and prepare flac data
cd ./egs2/mini_an4/asr1 || exit
rm -rf exp data dump
./run.sh --stage 1 --stop-stage 1
./run.sh --stage 2 --stop-stage 4 --feats-type "raw"

# Now we have flac files under dump/org/train_*/data/format.*/
# and wav.scp files under dump/train_*/

# [ESPnet Easy] test asr recipe with coverage
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    --task asr \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path conf/train_asr_transformer_debug.yaml \
    --train_sentencepiece_model \
    --run_collect_stats \
    --run_train

# finetuning
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
    --task asr \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path conf/train_asr_transformer_debug.yaml \
    --run_finetune

# Remove generated files in order to reduce the disk usage
rm -rf exp data/spm

# [ESPnet Easy] test streaming asr recipe with coverage
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    --task asr \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path conf/train_asr_streaming_debug.yaml \
    --train_sentencepiece_model \
    --run_collect_stats \
    --run_train

# finetuning
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
    --task asr \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path conf/train_asr_streaming_debug.yaml \
    --run_finetune

# Remove generated files in order to reduce the disk usage
rm -rf exp data/spm

# [ESPnet Easy] test asr transducer recipe with coverage
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    --task asr \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path conf/train_asr_transducer_debug.yaml \
    --train_sentencepiece_model \
    --run_collect_stats \
    --run_train

# finetuning
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
    --task asr \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path conf/train_asr_transducer_debug.yaml \
    --run_finetune

# Remove generated files in order to reduce the disk usage
rm -rf exp data/spm

# [ESPnet Easy] test conformer RNNT recipe with coverage
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    --task asr_transducer \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path conf/train_asr_conformer_rnnt_debug.yaml \
    --train_sentencepiece_model \
    --run_collect_stats \
    --run_train

# finetuning
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
    --task asr_transducer \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path conf/train_asr_conformer_rnnt_debug.yaml \
    --run_finetune

# Remove generated files in order to reduce the disk usage
rm -rf exp data/spm

# [ESPnet Easy] test lm recipe with coverage
cd ${cwd}/egs2/mini_an4/lm1
ln -sf ../asr1/data data
ln -sf ../asr1/dump dump
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    --task lm \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path conf/train_transformer.yaml \
    --train_sentencepiece_model \
    --run_collect_stats \
    --run_train

# finetune
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
    --task lm \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path conf/train_transformer.yaml \
    --run_finetune

# Remove generated files in order to reduce the disk usage
rm -rf exp data/spm


# [ESPnet Easy] test slu recipe with coverage
cd ${cwd}/egs2/mini_an4/s2t1
ln -sf ../asr1/data data
ln -sf ../asr1/dump dump
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    --task slu \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path conf/train_slu_transformer.yaml \
    --train_sentencepiece_model \
    --run_collect_stats \
    --run_train

# finetune
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
    --task slu \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path conf/train_slu_transformer.yaml \
    --run_finetune

# Remove generated files in order to reduce the disk usage
rm -rf exp data/spm


# [ESPnet Easy] test tts recipe with coverage
cd ${cwd}/egs2/mini_an4/tts1
ln -sf ../asr1/data data
ln -sf ../asr1/dump dump
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    --task tts \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path ../tts1/conf/train_tacotron2_debug.yaml \
    --train_sentencepiece_model \
    --run_collect_stats \
    --run_train

# finetune
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
    --task tts \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path ../tts1/conf/train_tacotron2_debug.yaml \
    --run_finetune

# Remove generated files in order to reduce the disk usage
rm -rf exp data/spm


# [ESPnet Easy] test gan-tts recipe with coverage
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    --task tts \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path conf/train_vits_debug.yaml \
    --train_sentencepiece_model \
    --run_collect_stats \
    --run_train

# finetune
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
    --task tts \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path conf/train_vits_debug.yaml \
    --run_finetune

# Remove generated files in order to reduce the disk usage
rm -rf exp data dump


# [ESPnet Easy] test asr2 recipe with coverage
cd ${cwd}/egs2/mini_an4/asr2
rm -rf exp data dump
gen_dummy_coverage
echo "==== [ESPnet2] ASR2 ==="
# data preparation
./run.sh --ngpu 0 --stage 1 --stop-stage 7 --use-lm false --asr-args "--num_workers 0"

python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    --task mt \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path conf/train_asr_transformer_debug.yaml \
    --run_collect_stats \
    --run_train

# finetune
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
    --task mt \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path conf/train_asr_transformer_debug.yaml \
    --run_finetune

# Remove generated files in order to reduce the disk usage
rm -rf exp dump data
cd "${cwd}"


# [ESPnet Easy] test enh recipe with coverage
if python -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.2.0")' &> /dev/null; then
    cd ${cwd}/egs2/mini_an4/enh1
    rm -rf exp data dump
    gen_dummy_coverage
    echo "==== [ESPnet2] ENH ==="
    ./run.sh --stage 1 --stop-stage 4 --python "${python}" --extra_wav_list "rirs.scp noises.scp"
    feats_types="raw"

    configs=("train_with_preprocessor_debug" "train_with_data_aug_debug" "train_debug")
    for conf in ${configs[@]}; do
        rm -rf exp
        python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
            --task enh \
            --data_path data \
            --train_dump_path dump/raw/train_nodev \
            --valid_dump_path dump/raw/train_dev \
            --exp_path ./exp \
            --config_path ./conf/${conf}.yaml \
            --train_sentencepiece_model \
            --run_collect_stats \
            --run_train

        python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
            --task enh \
            --data_path data \
            --train_dump_path dump/raw/train_nodev \
            --valid_dump_path dump/raw/train_dev \
            --exp_path ./exp \
            --config_path ./conf/${conf}.yaml \
            --run_finetune
    done
    rm -rf data exp dump

    # prepare for dynamic mixing
    ./run.sh --stage 1 --stop-stage 4 --python "${python}" --ref-num 2
    python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
        --task enh \
        --data_path data \
        --train_dump_path dump/raw/train_nodev \
        --valid_dump_path dump/raw/train_dev \
        --exp_path ./exp \
        --config_path ./conf/train_with_dynamic_mixing_debug.yaml \
        --train_sentencepiece_model \
        --run_collect_stats \
        --run_train

    python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
        --task enh \
        --data_path data \
        --train_dump_path dump/raw/train_nodev \
        --valid_dump_path dump/raw/train_dev \
        --exp_path ./exp \
        --config_path ./conf/train_with_dynamic_mixing_debug.yaml \
        --run_finetune

    cd "${cwd}"
fi

# # [ESPnet Easy] test enh-tse recipe with coverage
if python -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.2.0")' &> /dev/null; then
    cd ${cwd}/egs2/mini_an4/tse1
    rm -rf exp data dump
    gen_dummy_coverage
    echo "==== [ESPnet2] ENH_TSE ==="
    feats_types="raw"

    # simple pattern
    ./run.sh --stage 1 --stop-stage 4 --ref-num 1
    python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
        --task enh_tse \
        --data_path data \
        --train_dump_path dump/raw/train_nodev \
        --valid_dump_path dump/raw/train_dev \
        --exp_path ./exp \
        --config_path ./conf/train_debug.yaml \
        --train_sentencepiece_model \
        --run_collect_stats \
        --run_train

    python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
        --task enh_tse \
        --data_path data \
        --train_dump_path dump/raw/train_nodev \
        --valid_dump_path dump/raw/train_dev \
        --exp_path ./exp \
        --config_path ./conf/train_debug.yaml \
        --run_finetune

    rm -rf exp dump data
fi

# [ESPnet Easy] test enh-asr recipe with coverage
if python -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.2.0")' &> /dev/null; then
    cd ${cwd}/egs2/mini_an4/enh_asr1
    rm -rf exp data dump
    gen_dummy_coverage
    echo "==== [ESPnet2] ENH_ASR ==="
    ./run.sh --ngpu 0 --stage 0 --stop-stage 4 --feats-type "raw" --spk-num 1 \
        --enh_asr_args "--enh_separator_conf num_spk=1 --num_workers 0"

    python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
        --task enh_s2t \
        --data_path data \
        --train_dump_path dump/raw/train_nodev \
        --valid_dump_path dump/raw/train_dev \
        --exp_path ./exp \
        --config_path ./conf/train_debug.yaml \
        --train_sentencepiece_model \
        --run_collect_stats \
        --run_train

    python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
        --task enh_s2t \
        --data_path data \
        --train_dump_path dump/raw/train_nodev \
        --valid_dump_path dump/raw/train_dev \
        --exp_path ./exp \
        --config_path ./conf/train_debug.yaml \
        --run_finetune

    # Remove generated files in order to reduce the disk usage
    rm -rf exp dump data
    cd "${cwd}"
fi

# [ESPnet Easy] test ssl recipe with coverage
if python3 -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.12.0")' &> /dev/null; then
    cd ${cwd}/egs2/mini_an4/ssl1
    rm -rf exp data dump
    gen_dummy_coverage
    echo "==== [ESPnet2] SSL1/HUBERT ==="
    ./run.sh --ngpu 0 --stage 1 --stop-stage 5 --feats-type "raw" \
        --token_type "word" --hubert-args "--num_workers 0" --train_stop_iter 0

    python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
        --task hubert \
        --data_path data \
        --train_dump_path dump/raw/train_nodev \
        --valid_dump_path dump/raw/train_dev \
        --exp_path ./exp \
        --config_path ./conf/train_ssl_torchaudiohubert_base_pretrain_it0_debug.yaml \
        --run_collect_stats \
        --run_train

    # It seems there is no inference class for HubertTask?
    # python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
    #     --task hubert \
    #     --data_path data \
    #     --train_dump_path dump/raw/train_nodev \
    #     --valid_dump_path dump/raw/train_dev \
    #     --exp_path ./exp \
    #     --config_path ./conf/train_ssl_torchaudiohubert_base_pretrain_it0_debug.yaml \
    #     --run_finetune

    # Remove generated files in order to reduce the disk usage
    rm -rf exp dump data
    cd "${cwd}"
fi

# [ESPnet Easy] test st recipe with coverage
cd ${cwd}/egs2/mini_an4/st1
echo "==== [ESPnet2] ST ==="
rm -rf exp data dump
./run.sh --stage 1 --stop-stage 5 --feats-type "raw" --tgt_token_type "bpe" --src_token_type "bpe"

python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    --task st \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path ./conf/train_st_debug.yaml \
    --train_sentencepiece_model \
    --run_collect_stats \
    --run_train

python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
    --task st \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path ./conf/train_st_debug.yaml \
    --run_finetune

# streaming
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    --task st \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path ./conf/train_st_streaming_debug.yaml \
    --train_sentencepiece_model \
    --run_collect_stats \
    --run_train

python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
    --task st \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path ./conf/train_st_streaming_debug.yaml \
    --run_finetune

# Remove generated files in order to reduce the disk usage
rm -rf exp dump data

# [ESPnet Easy] test s2t1 recipe with coverage
cd ${cwd}/egs2/mini_an4/s2t1
rm -rf exp dump data
gen_dummy_coverage
echo "==== [ESPnet2] S2T1 ==="
./run.sh --ngpu 0 --stage 1 --stop_stage 4 --feats_type raw --audio_format flac.ark --token_type bpe
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    --task s2t \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path ./conf/train_transformer.yaml \
    --train_sentencepiece_model \
    --run_collect_stats \
    --run_train

python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
    --task s2t \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path ./conf/train_transformer.yaml \
    --run_finetune

# Remove generated files in order to reduce the disk usage
rm -rf exp dump data
cd "${cwd}"

cd "${cwd}" || exit


echo "=== report ==="
python -m coverage combine egs2/*/*/.coverage
python -m coverage report
python -m coverage xml
