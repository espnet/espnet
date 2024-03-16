#!/usr/bin/env bash

set -euo pipefail

source tools/activate_python.sh
PYTHONPATH="${PYTHONPATH:-}:$(pwd)/tools/s3prl"
export PYTHONPATH
# python="python -m coverage run --append"
cwd=$(pwd)
tasks=("asr" "asr_transducer" "gan_tts" "hubert" "lm" "s2t" "slu" "st" "tts" "uasr")

gen_dummy_coverage(){
    # To avoid a problem when parallel running for `coverage run`.
    # Please put this command after cd ./egs2/foo/bar
    touch empty.py; ${python} empty.py
}

#### Make sure chainer-independent ####
python3 -m pip uninstall -y chainer

# Download mini_an4 as test data and prepare flac data
cd ./egs2/mini_an4/asr1
./run.sh --stage 1 --stop-stage 1
./run.sh --stage 2 --stop-stage 4 --feats-type "raw"
cp -r dump/raw data/
./run.sh --stage 2 --stop-stage 4 --feats-type "raw_copy" \
    --train_set raw/train_nodev --valid_set raw/train_dev --test_sets raw/test

# Now we have flac files under dump/org/train_*/data/format.*/
# and wav.scp files under dump/train_*/

# [ESPnet Easy] test asr recipe with coverage
python -m coverage run --append ../../../ci/test_integrate_easy.py \
    --task asr \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path conf/train_asr_transformer_debug.yaml \
    --train_sentencepiece_model \
    --run_collect_stats \
    --run_train

# Remove generated files in order to reduce the disk usage
rm -rf exp data/spm

# [ESPnet Easy] test asr transducer recipe with coverage
python -m coverage run --append ../../../ci/test_integrate_easy.py \
    --task asr \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path conf/train_asr_transducer_debug.yaml \
    --train_sentencepiece_model \
    --run_collect_stats \
    --run_train

# Remove generated files in order to reduce the disk usage
rm -rf exp data/spm

# [ESPnet Easy] test lm recipe with coverage
python -m coverage run --append ../../../ci/test_integrate_easy.py \
    --task lm \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path ../lm1/conf/train_transformer.yaml \
    --train_sentencepiece_model \
    --run_collect_stats \
    --run_train 

# Remove generated files in order to reduce the disk usage
rm -rf exp data/spm


# [ESPnet Easy] test slu recipe with coverage
python -m coverage run --append ../../../ci/test_integrate_easy.py \
    --task slu \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path ../s2t1/conf/train_slu_transformer.yaml \
    --train_sentencepiece_model \
    --run_collect_stats \
    --run_train 

# Remove generated files in order to reduce the disk usage
rm -rf exp data/spm


# [ESPnet Easy] test tts recipe with coverage
python -m coverage run --append ../../../ci/test_integrate_easy.py \
    --task tts \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path ../tts1/conf/train_tacotron2_debug.yaml \
    --train_sentencepiece_model \
    --run_collect_stats \
    --run_train 

# Remove generated files in order to reduce the disk usage
rm -rf exp data/spm

cd "${cwd}"


echo "=== report ==="
python -m coverage combine egs2/*/*/.coverage
python -m coverage report
python -m coverage xml
