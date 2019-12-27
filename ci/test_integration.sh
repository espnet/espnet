#!/usr/bin/env bash

# test asr recipe
cwd=$(pwd)
cd ./egs/mini_an4/asr1 || exit 1
. path.sh  # source here to avoid undefined variable errors

set -euo pipefail

echo "==== ASR (backend=pytorch) ==="
./run.sh
echo "==== ASR (backend=pytorch, dtype=float64) ==="
#./run.sh --stage 3 --train-config "$(change_yaml.py conf/train.yaml -a train-dtype=float64)" --decode-config "$(change_yaml.py conf/decode.yaml -a api=v2 -a dtype=float64)"
echo "==== ASR (backend=chainer) ==="
#./run.sh --stage 3 --backend chainer
echo "==== ASR (backend=pytorch num-encs 2) ==="
#./run.sh --stage 2 --train-config ./conf/train_mulenc2.yaml --decode-config ./conf/decode_mulenc2.yaml --mulenc true

# test transducer recipe
echo "=== ASR (backend=pytorch, model=rnnt) ==="
#./run.sh --stage 3 --train-config conf/train_transducer.yaml \
#         --decode-config conf/decode_transducer.yaml
echo "=== ASR (backend=pytorch, model=transformer-transducer) ==="
./run.sh --stage 3 --train-config conf/train_transformer_transducer.yaml \
         --decode-config conf/decode_transducer.yaml

cd ${cwd} || exit 1

# test asr_mix recipe
cd ./egs/mini_an4/asr_mix1 || exit 1
echo "==== ASR Mix (backend=pytorch) ==="
./run.sh
cd ${cwd} || exit 1

# test tts recipe
cd ./egs/mini_an4/tts1 || exit 1
echo "==== TTS (backend=pytorch) ==="
./run.sh
cd ${cwd} || exit 1

# TODO(karita): test mt, st?
