#!/usr/bin/env bash

python="coverage run --append"

touch .coverage

# test asr recipe
cwd=$(pwd)
cd ./egs/mini_an4/asr1 || exit 1
ln -sf ${cwd}/.coverage .
. path.sh  # source here to avoid undefined variable errors

set -euo pipefail

echo "==== ASR (backend=pytorch lm=RNNLM) ==="
./run.sh --python "${python}"
echo "==== ASR (backend=pytorch, lm=TransformerLM) ==="
./run.sh --python "${python}" --stage 3 --stop-stage 3 --lm-config conf/lm_transformer.yaml --decode-config "$(change_yaml.py conf/decode.yaml -a api=v2)"
# skip duplicated ASR training stage 4
./run.sh --python "${python}" --stage 5 --lm-config conf/lm_transformer.yaml --decode-config "$(change_yaml.py conf/decode.yaml -a api=v2)"
echo "==== ASR (backend=pytorch, dtype=float64) ==="
./run.sh --python "${python}" --stage 3 --train-config "$(change_yaml.py conf/train.yaml -a train-dtype=float64)" --decode-config "$(change_yaml.py conf/decode.yaml -a api=v2 -a dtype=float64)"
echo "==== ASR (backend=pytorch, quantize-asr-model true, quantize-lm-model true) ==="
./run.sh --python "${python}" --stage 5 --decode-config "$(change_yaml.py conf/decode.yaml -a quantize-asr-model=true -a quantize-lm-model=true)"
echo "==== ASR (backend=pytorch, quantize-asr-model true, quantize-lm-model true api v2) ==="
./run.sh --python "${python}" --stage 5 --decode-config "$(change_yaml.py conf/decode.yaml -a quantize-asr-model=true -a quantize-lm-model=true -a quantize-config=['Linear'] -a api=v2)"

echo "==== ASR (backend=chainer) ==="
./run.sh --python "${python}" --stage 3 --backend chainer

# skip duplicated ASR training stage 2,3
# test rnn recipe
echo "=== ASR (backend=pytorch, model=rnn-pure-ctc) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_pure_ctc.yaml \
        --decode-config conf/decode_pure_ctc.yaml
echo "=== ASR (backend=pytorch, model=rnn-no-ctc) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_no_ctc.yaml \
        --decode-config conf/decode_no_ctc.yaml

# test transformer recipe
echo "=== ASR (backend=pytorch, model=transformer) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_transformer.yaml \
        --decode-config conf/decode.yaml
./run.sh --python "${python}" --stage 5 --train-config conf/train_transformer.yaml \
        --decode-config conf/decode.yaml --metric acc
./run.sh --python "${python}" --stage 5 --train-config conf/train_transformer.yaml \
        --decode-config conf/decode.yaml --metric loss
echo "=== ASR (backend=pytorch, model=conformer) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_conformer.yaml \
        --decode-config conf/decode.yaml
echo "=== ASR (backend=pytorch, model=transformer-pure-ctc) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_transformer_pure_ctc.yaml \
        --decode-config conf/decode_pure_ctc.yaml
echo "=== ASR (backend=pytorch, model=conformer-pure-ctc) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_conformer_pure_ctc.yaml \
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
ln -sf ${cwd}/.coverage .

echo "==== ASR Mix (backend=pytorch, model=rnn) ==="
./run.sh --python "${python}" --train-config conf/train_multispkr.yaml
echo "==== ASR Mix (backend=pytorch, model=transformer) ==="
./run.sh --python "${python}" --stage 4 --train-config conf/train_multispkr_transformer.yaml
# Remove generated files in order to reduce the disk usage
rm -rf exp tensorboard dump data
cd "${cwd}" || exit 1

# test st recipe
cd ./egs/mini_an4/st1 || exit 1
ln -sf ${cwd}/.coverage .

echo "==== ST (backend=pytorch) ==="
./run.sh --python "${python}"
echo "==== ST (backend=pytorch asr0.3) ==="
./run.sh --python "${python}" --stage 4 --train_config conf/train_asr0.3.yaml
echo "==== ST (backend=pytorch ctc asr0.3) ==="
./run.sh --python "${python}" --stage 4 --train_config conf/train_ctc_asr0.3.yaml
echo "==== ST (backend=pytorch mt0.3) ==="
./run.sh --python "${python}" --stage 4 --train_config conf/train_mt0.3.yaml
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
./run.sh --python "${python}" --stage 4 --train_config conf/train_transformer_asr0.3.yaml
echo "==== ST (backend=pytorch ctc asr0.3, model=transformer) ==="
./run.sh --python "${python}" --stage 4 --train_config conf/train_transformer_ctc_asr0.3.yaml
echo "==== ST (backend=pytorch mt0.3, model=transformer) ==="
./run.sh --python "${python}" --stage 4 --train_config conf/train_transformer_mt0.3.yaml
echo "==== ST (backend=pytorch asr0.2 mt0.2, model=transformer) ==="
./run.sh --python "${python}" --stage 4 --train_config conf/train_transformer_asr0.2_mt0.2.yaml
echo "==== ST (backend=pytorch asr0.2 mt0.2, model=conformer) ==="
./run.sh --python "${python}" --stage 4 --train_config conf/train_conformer_asr0.2_mt0.2.yaml
# Remove generated files in order to reduce the disk usage
rm -rf exp tensorboard dump data
cd "${cwd}" || exit 1

# test mt recipe
cd ./egs/mini_an4/mt1 || exit 1
ln -sf ${cwd}/.coverage .

echo "==== MT (backend=pytorch) ==="
./run.sh --python "${python}"
echo "==== MT (backend=pytorch, model=transformer) ==="
./run.sh --python "${python}" --stage 4 --train_config conf/train_transformer.yaml
./run.sh --python "${python}" --stage 5 --train_config conf/train_transformer.yaml \
    --metric acc
./run.sh --python "${python}" --stage 5 --train_config conf/train_transformer.yaml \
    --metric bleu
./run.sh --python "${python}" --stage 5 --train_config conf/train_transformer.yaml \
    --metric loss
# Remove generated files in order to reduce the disk usage
rm -rf exp tensorboard dump data
cd "${cwd}" || exit 1

# test tts recipe
cd ./egs/mini_an4/tts1 || exit 1
ln -sf ${cwd}/.coverage .

echo "==== TTS (backend=pytorch) ==="
./run.sh --python "${python}"
# Remove generated files in order to reduce the disk usage
rm -rf exp tensorboard dump data
cd "${cwd}" || exit 1

echo "=== run integration tests at test_utils ==="

PATH=$(pwd)/bats-core/bin:$PATH
if ! [ -x "$(command -v bats)" ]; then
    echo "=== install bats ==="
    git clone https://github.com/bats-core/bats-core.git
fi
bats test_utils/integration_test_*.bats


#### Make sure chainer-independent ####
python3 -m pip uninstall -y chainer

# [ESPnet2] test asr recipe
cd ./egs2/mini_an4/asr1 || exit 1
ln -sf ${cwd}/.coverage .
echo "==== [ESPnet2] ASR ==="
./run.sh --stage 1 --stop-stage 1
feats_types="raw fbank_pitch"
token_types="bpe char"
for t in ${feats_types}; do
    ./run.sh --stage 2 --stop-stage 4 --feats-type "${t}" --python "${python}"
done
for t in ${token_types}; do
    ./run.sh --stage 5 --stop-stage 5 --token-type "${t}" --python "${python}"
done
for t in ${feats_types}; do
    for t2 in ${token_types}; do
        echo "==== feats_type=${t}, token_types=${t2} ==="
        ./run.sh --ngpu 0 --stage 6 --stop-stage 13 --skip-upload false --feats-type "${t}" --token-type "${t2}" \
            --asr-args "--max_epoch=1" --lm-args "--max_epoch=1" --python "${python}"
    done
done
# Remove generated files in order to reduce the disk usage
rm -rf exp dump data
cd "${cwd}" || exit 1

# [ESPnet2] test tts recipe
cd ./egs2/mini_an4/tts1 || exit 1
ln -sf ${cwd}/.coverage .
echo "==== [ESPnet2] TTS ==="
./run.sh --stage 1 --stop-stage 1 --python "${python}"
feats_types="raw fbank stft"
for t in ${feats_types}; do
    echo "==== feats_type=${t} ==="
    ./run.sh --ngpu 0 --stage 2 --stop-stage 8 --skip-upload false --feats-type "${t}" --train-args "--max_epoch 1" --python "${python}"
done
# Remove generated files in order to reduce the disk usage
rm -rf exp dump data
cd "${cwd}" || exit 1

# [ESPnet2] test enh recipe
if python -c 'import torch as t; from distutils.version import LooseVersion as L; assert L(t.__version__) >= L("1.2.0")' &> /dev/null;  then
    cd ./egs2/mini_an4/enh1 || exit 1
    ln -sf ${cwd}/.coverage .
    echo "==== [ESPnet2] ENH ==="
    ./run.sh --stage 1 --stop-stage 1 --python "${python}"
    feats_types="raw"
    for t in ${feats_types}; do
        echo "==== feats_type=${t} ==="
        ./run.sh --ngpu 0 --stage 2 --stop-stage 10 --skip-upload false --feats-type "${t}" --spk-num 1 --enh-args "--max_epoch=1" --python "${python}"
    done
    # Remove generated files in order to reduce the disk usage
    rm -rf exp dump data
    cd "${cwd}" || exit 1
fi

# [ESPnet2] Validate configuration files
echo "<blank>" > dummy_token_list
echo "==== [ESPnet2] Validation configuration files ==="
if python3 -c 'import torch as t; from distutils.version import LooseVersion as L; assert L(t.__version__) >= L("1.6.0")' &> /dev/null;  then
    for f in egs2/*/asr1/conf/train_asr*.yaml; do
        python3 -m espnet2.bin.asr_train --config "${f}" --iterator_type none --dry_run true --output_dir out --token_list dummy_token_list
    done
    for f in egs2/*/asr1/conf/train_lm*.yaml; do
        python3 -m espnet2.bin.lm_train --config "${f}" --iterator_type none --dry_run true --output_dir out --token_list dummy_token_list
    done
    for f in egs2/*/tts1/conf/train*.yaml; do
        python3 -m espnet2.bin.tts_train --config "${f}" --iterator_type none --normalize none --dry_run true --output_dir out --token_list dummy_token_list
    done
    for f in egs2/*/enh1/conf/train*.yaml; do
        python -m espnet2.bin.enh_train --config "${f}" --iterator_type none --dry_run true --output_dir out
    done
fi

# These files must be same each other.
for base in cmd.sh conf/slurm.conf conf/queue.conf conf/pbs.conf; do
    file1=
    for f in egs2/*/*/"${base}"; do
        if [ -z "${file1}" ]; then
            file1="${f}"
        fi
        diff "${file1}" "${f}" || { echo "Error: ${file1} and ${f} differ: To solve: for f in egs2/*/*/${base}; do cp egs2/TEMPLATE/asr1/${base} \${f}; done" ; exit 1; }
    done
done


echo "==== [ESPnet2] test setup.sh ==="
for d in egs2/TEMPLATE/*; do
    if [ -d "${d}" ]; then
        d="${d##*/}"
        egs2/TEMPLATE/"$d"/setup.sh egs2/test/"${d}"
    fi
done
echo "=== report ==="

coverage report
coverage xml
