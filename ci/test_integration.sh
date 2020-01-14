#!/usr/bin/env bash

# test asr recipe
cwd=$(pwd)
cd ./egs/mini_an4/asr1 || exit 1
. path.sh  # source here to avoid undefined variable errors

set -euo pipefail

echo "==== ASR (backend=pytorch) ==="
./run.sh
echo "==== ASR (backend=pytorch, dtype=float64) ==="
./run.sh --stage 3 --train-config "$(change_yaml.py conf/train.yaml -a train-dtype=float64)" --decode-config "$(change_yaml.py conf/decode.yaml -a api=v2 -a dtype=float64)"
echo "==== ASR (backend=chainer) ==="
./run.sh --stage 3 --backend chainer
echo "==== ASR (backend=pytorch num-encs 2) ==="
./run.sh --stage 2 --train-config ./conf/train_mulenc2.yaml --decode-config ./conf/decode_mulenc2.yaml --mulenc true
cd "${cwd}" || exit 1

# test asr_mix recipe
cd ./egs/mini_an4/asr_mix1 || exit 1
echo "==== ASR Mix (backend=pytorch) ==="
./run.sh
cd "${cwd}" || exit 1

# test tts recipe
cd ./egs/mini_an4/tts1 || exit 1
echo "==== TTS (backend=pytorch) ==="
./run.sh
cd "${cwd}" || exit 1

# [ESPnet2] test asr recipe
cd ./egs2/mini_an4/asr1 || exit 1
echo "==== [ESPnet2] ASR ==="
./run.sh --stage 1 --stop-stage 1
feats_types="raw fbank_pitch"
token_types="bpe char"
for t in ${feats_types}; do
    ./run.sh --stage 2 --stop-stage 2 --feats-type "${t}"
done
for t in ${token_types}; do
    ./run.sh --stage 3 --stop-stage 3 --token-type "${t}"
done
for t in ${feats_types}; do
    for t2 in ${token_types}; do
        echo "==== feats_type=${t}, token_types=${t2} ==="
        ./run.sh --ngpu 0 --stage 4 --stop-stage 100 --feats-type "${t}" --token-type "${t2}" \
            --asr-args "--max_epoch=1" --lm-args "--max_epoch=1"
    done
done
cd "${cwd}" || exit 1

# [ESPnet2] test tts recipe
cd ./egs2/mini_an4/tts1 || exit 1
echo "==== [ESPnet2] TTS ==="
./run.sh --stage 1 --stop-stage 1
feats_types="raw fbank stft"
for t in ${feats_types}; do
    echo "==== feats_type=${t} ==="
    ./run.sh --stage 2 --stop-stage 100 --feats-type "${t}" --train-args "--max_epoch 1"
done
cd "${cwd}" || exit 1


# [ESPnet2] Validate configuration files
echo "==== [ESPnet2] Validation configuration files ==="
if python -c 'import torch as t; from distutils.version import LooseVersion as L; assert L(t.__version__) >= L("1.1.0")' &> /dev/null;  then
    for f in egs2/*/asr1/conf/train_asr*.yaml; do
        python -m espnet2.bin.asr_train --config "${f}" --print_config
    done
    for f in egs2/*/asr1/conf/train_lm*.yaml; do
        python -m espnet2.bin.lm_train --config "${f}" --print_config
    done
    for f in egs2/*/tts1/conf/train_*.yaml; do
        python -m espnet2.bin.tts_train --config "${f}" --print_config
    done
fi


# TODO(karita): test mt, st?
