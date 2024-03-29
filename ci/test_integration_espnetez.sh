#!/usr/bin/env bash

# set -euo pipefail

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
./run.sh --stage 1 --stop-stage 1
./run.sh --stage 2 --stop-stage 4 --feats-type "raw"

# Now we have flac files under dump/org/train_*/data/format.*/
# and wav.scp files under dump/train_*/

rm -rf exp data/spm
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

if python3 -c "from warprnnt_pytorch import RNNTLoss" &> /dev/null; then
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
fi

# Remove generated files in order to reduce the disk usage
rm -rf exp data/spm

# [ESPnet Easy] test lm recipe with coverage
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    --task lm \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path ../lm1/conf/train_transformer.yaml \
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
    --config_path ../lm1/conf/train_transformer.yaml \
    --run_finetune

# Remove generated files in order to reduce the disk usage
rm -rf exp data/spm


# [ESPnet Easy] test slu recipe with coverage
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    --task slu \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path ../s2t1/conf/train_slu_transformer.yaml \
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
    --config_path ../s2t1/conf/train_slu_transformer.yaml \
    --run_finetune

# Remove generated files in order to reduce the disk usage
rm -rf exp data/spm


# [ESPnet Easy] test tts recipe with coverage
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

cd "${cwd}" || exit


echo "=== report ==="
python -m coverage combine egs2/*/*/.coverage
python -m coverage report
python -m coverage xml
