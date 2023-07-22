#!/usr/bin/env bash

python="coverage run --append"

cwd=$(pwd)

# test asr recipe
cd ./egs/mini_an4/asr1 || exit 1
. path.sh  # source here to avoid undefined variable errors

set -euo pipefail

echo "==== ASR (backend=chainer) ==="
./run.sh --python "${python}" --backend chainer
echo "==== ASR (backend=pytorch lm=RNNLM) ==="
./run.sh --python "${python}" --stage 3
echo "==== ASR (backend=pytorch, lm=TransformerLM) ==="
./run.sh --python "${python}" --stage 3 --stop-stage 3 --lm-config conf/lm_transformer.yaml --decode-config "$(change_yaml.py conf/decode.yaml -a api=v2)"
echo "==== ASR (backend=pytorch, lm=TransformerLM, api v2) ==="
./run.sh --python "${python}" --stage 5 --lm-config conf/lm_transformer.yaml --decode-config "$(change_yaml.py conf/decode.yaml -a api=v2)"
echo "==== ASR (backend=pytorch, dtype=float64) ==="
./run.sh --python "${python}" --stage 3 --train-config "$(change_yaml.py conf/train.yaml -a train-dtype=float64)" --decode-config "$(change_yaml.py conf/decode.yaml -a api=v2 -a dtype=float64)"
echo "==== ASR (backend=pytorch, quantize-asr-model true, quantize-lm-model true) ==="
./run.sh --python "${python}" --stage 5 --decode-config "$(change_yaml.py conf/decode.yaml -a quantize-asr-model=true -a quantize-lm-model=true)"
echo "==== ASR (backend=pytorch, quantize-asr-model true, quantize-lm-model true api v2) ==="
./run.sh --python "${python}" --stage 5 --decode-config "$(change_yaml.py conf/decode.yaml -a quantize-asr-model=true -a quantize-lm-model=true -a quantize-config=['Linear'] -a api=v2)"

# this is used for the transfer learning
echo "==== ASR (backend=pytorch, model=rnn-pure-ctc) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_pure_ctc.yaml --decode-config conf/decode_pure_ctc.yaml

# test transformer recipe
echo "=== ASR (backend=pytorch, model=transformer) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_transformer.yaml \
        --decode-config conf/decode.yaml
echo "=== ASR (backend=pytorch, model=conformer) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_conformer.yaml \
        --decode-config conf/decode.yaml
echo "=== ASR (backend=pytorch, model=transformer-pure-ctc) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_transformer_pure_ctc.yaml \
        --decode-config conf/decode_pure_ctc.yaml
echo "=== ASR (backend=pytorch, model=transformer-no-ctc) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_transformer_no_ctc.yaml \
        --decode-config conf/decode_no_ctc.yaml
echo "=== ASR (backend=pytorch num-encs 2, model=transformer) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_transformer.yaml \
        --decode-config conf/decode.yaml

# test transducer recipe
echo "=== ASR (backend=pytorch, model=rnnt) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_transducer.yaml \
        --decode-config conf/decode_transducer.yaml
echo "=== ASR (backend=pytorch, model=transformer-transducer) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_transformer_transducer.yaml \
        --decode-config conf/decode_transducer.yaml
echo "=== ASR (backend=pytorch, model=conformer-transducer) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_conformer_transducer.yaml \
        --decode-config conf/decode_transducer.yaml

# test transducer with auxiliary task recipe
echo "=== ASR (backend=pytorch, model=rnnt, tasks=L1+L2+L3+L4+L5)"
./run.sh --python "${python}" --stage 4 --train-config conf/train_transducer_aux.yaml \
         --decode-config conf/decode_transducer.yaml

# test finetuning
## test transfer learning
echo "=== ASR (backend=pytorch, model=rnnt, transfer_learning=enc) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_transducer_pre_init_enc.yaml \
         --decode-config conf/decode_transducer.yaml
echo "=== ASR (backend=pytorch, model=rnnt, transfer_learning=LM) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_transducer_pre_init_lm.yaml \
         --decode-config conf/decode_transducer.yaml
## to do: cover all tasks + freezing option

echo "==== ASR (backend=pytorch num-encs 2) ==="
./run.sh --python "${python}" --stage 2 --train-config ./conf/train_mulenc2.yaml --decode-config ./conf/decode_mulenc2.yaml --mulenc true
# Remove generated files in order to reduce the disk usage
rm -rf exp tensorboard dump data
cd ${cwd} || exit 1

# test asr_mix recipe
cd ./egs/mini_an4/asr_mix1 || exit 1

echo "==== ASR Mix (backend=pytorch, model=rnn) ==="
./run.sh --python "${python}" --train-config conf/train_multispkr.yaml
echo "==== ASR Mix (backend=pytorch, model=transformer) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_multispkr_transformer.yaml
# Remove generated files in order to reduce the disk usage
rm -rf exp tensorboard dump data
cd "${cwd}" || exit 1

# test st recipe
cd ./egs/mini_an4/st1 || exit 1

echo "==== ST (backend=pytorch) ==="
./run.sh --python "${python}"
echo "==== ST (backend=pytorch ctc asr0.3) ==="
./run.sh --python "${python}" --stage 4 --train_config conf/train_ctc_asr0.3.yaml
echo "==== ST (backend=pytorch asr0.2 mt0.2) ==="
./run.sh --python "${python}" --stage 4 --train_config conf/train_asr0.2_mt0.2.yaml
echo "==== ST (backend=pytorch, model=transformer) ==="
./run.sh --python "${python}" --stage 4 --train_config conf/train_transformer.yaml
./run.sh --python "${python}" --stage 5 --train_config conf/train_transformer.yaml \
    --metric acc
./run.sh --python "${python}" --stage 5 --train_config conf/train_transformer.yaml \
    --metric bleu
./run.sh --python "${python}" --stage 5 --train_config conf/train_transformer.yaml \
    --metric loss
echo "==== ST (backend=pytorch asr0.3, model=transformer) ==="
./run.sh --python "${python}" --stage 4 --train_config conf/train_transformer_ctc_asr0.3.yaml
echo "==== ST (backend=pytorch asr0.2 mt0.2, model=conformer) ==="
./run.sh --python "${python}" --stage 4 --train_config conf/train_conformer_asr0.2_mt0.2.yaml
# Remove generated files in order to reduce the disk usage
rm -rf exp tensorboard dump data
cd "${cwd}" || exit 1

# test mt recipe
cd ./egs/mini_an4/mt1 || exit 1

echo "==== MT (backend=pytorch) ==="
./run.sh --python "${python}"
echo "==== MT (backend=pytorch, model=transformer) ==="
./run.sh --python "${python}" --stage 4 --train_config conf/train_transformer.yaml
# Remove generated files in order to reduce the disk usage
rm -rf exp tensorboard dump data
cd "${cwd}" || exit 1

# test tts recipe
cd ./egs/mini_an4/tts1 || exit 1

echo "==== TTS (backend=pytorch) ==="
./run.sh --python "${python}"
# Remove generated files in order to reduce the disk usage
rm -rf exp tensorboard dump data
cd "${cwd}" || exit 1

echo "=== report ==="

coverage combine egs/*/*/.coverage
coverage report
coverage xml
