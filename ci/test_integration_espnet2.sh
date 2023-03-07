#!/usr/bin/env bash

set -euo pipefail

source tools/activate_python.sh
PYTHONPATH="${PYTHONPATH:-}:$(pwd)/tools/s3prl"
export PYTHONPATH
python="coverage run --append"
cwd=$(pwd)

gen_dummy_coverage(){
    # To avoid a problem when parallel running for `coverage run`.
    # Please put this command after cd ./egs2/foo/bar
    touch empty.py; ${python} empty.py
}

#### Make sure chainer-independent ####
python3 -m pip uninstall -y chainer

# [ESPnet2] test asr recipe
cd ./egs2/mini_an4/asr1
gen_dummy_coverage
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
            --asr-args "--max_epoch=1 --decoder rnn" --lm-args "--max_epoch=1" --python "${python}"
    done
done
echo "==== feats_type=raw, token_types=bpe, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn ==="
./run.sh --ngpu 0 --stage 10 --stop-stage 13 --skip-upload false --feats-type "raw" --token-type "bpe" \
    --feats_normalize "utterance_mvn" --lm-args "--max_epoch=1" --python "${python}" \
    --asr-args "--model_conf extract_feats_in_collect_stats=false --max_epoch=1 --decoder=rnn"

echo "==== use_streaming, feats_type=raw, token_types=bpe, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn ==="
./run.sh --use_streaming true --ngpu 0 --stage 6 --stop-stage 13 --skip-upload false --feats-type "raw" --token-type "bpe" \
    --feats_normalize "utterance_mvn" --lm-args "--max_epoch=1" --python "${python}" \
    --asr-args "--model_conf extract_feats_in_collect_stats=false --max_epoch=1 --encoder=contextual_block_transformer --decoder=transformer
                --encoder_conf block_size=40 --encoder_conf hop_size=16 --encoder_conf look_ahead=16"

if python3 -c "import k2" &> /dev/null; then
    echo "==== use_k2, num_paths > nll_batch_size, feats_type=raw, token_types=bpe, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn ==="
    ./run.sh --num_paths 500 --nll_batch_size 20 --use_k2 true --ngpu 0 --stage 12 --stop-stage 13 --skip-upload false --feats-type "raw" --token-type "bpe" \
        --feats_normalize "utterance_mvn" --lm-args "--max_epoch=1" --python "${python}" \
        --asr-args "--model_conf extract_feats_in_collect_stats=false --max_epoch=1 --decoder=rnn"

    echo "==== use_k2, num_paths == nll_batch_size, feats_type=raw, token_types=bpe, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn ==="
    ./run.sh --num_paths 20 --nll_batch_size 20 --use_k2 true --ngpu 0 --stage 12 --stop-stage 13 --skip-upload false --feats-type "raw" --token-type "bpe" \
       --feats_normalize "utterance_mvn" --lm-args "--max_epoch=1" --python "${python}" \
       --asr-args "--model_conf extract_feats_in_collect_stats=false --max_epoch=1 --decoder=rnn"
fi

if python3 -c "from warprnnt_pytorch import RNNTLoss" &> /dev/null; then
    echo "==== [ESPnet2] ASR Transducer (standalone) ==="

    for t in ${token_types}; do
        asr_tag="transducer_${t}"

        echo "==== [Conformer-RNN-T] feats_type=raw, token_types=${t}, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn ==="
        ./run.sh --asr_task "asr_transducer" --ngpu 0 --stage 10 --stop-stage 13 --skip-upload false --feats-type "raw" --token-type ${t} \
            --feats_normalize "utterance_mvn" --lm-args "--max_epoch=1" --python "${python}" --inference_asr_model "valid.loss.best.pth" \
            --asr-tag "${asr_tag}_conformer" --asr-args "--model_conf extract_feats_in_collect_stats=false --max_epoch=1 \
            --encoder_conf body_conf='[{'block_type': 'conformer', 'hidden_size': 30, 'linear_size': 30, 'heads': 2, 'conv_mod_kernel_size': 3}]' \
            --decoder_conf='{'embed_size': 30, 'hidden_size': 30}' --joint_network_conf joint_space_size=30"

        echo "==== [Streaming Conformer-RNN-T] feats_type=raw, token_types=${t}, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn ==="
        ./run.sh --asr_task "asr_transducer" --ngpu 0 --stage 10 --stop-stage 13 --skip-upload false --feats-type "raw" --token-type ${t} \
            --feats_normalize "utterance_mvn" --lm-args "--max_epoch=1" --python "${python}" --inference_asr_model "valid.loss.best.pth" \
            --asr-tag "${asr_tag}_conformer_streaming" --asr-args "--model_conf extract_feats_in_collect_stats=false --max_epoch=1 \
            --encoder_conf main_conf='{'dynamic_chunk_training': True}' \
            --encoder_conf body_conf='[{'block_type': 'conformer', 'hidden_size': 30, 'linear_size': 30, 'heads': 2, 'conv_mod_kernel_size': 3}]' \
            --decoder_conf='{'embed_size': 30, 'hidden_size': 30}' --joint_network_conf joint_space_size=30 " \
            --inference-args "--streaming true --chunk_size 2 --left_context 2 --right_context 0"
    done
fi

echo "==== [PIT_ASR] feats_type=raw, token_types=bpe, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn ==="
for i in $(seq 2); do
    cp dump/raw/train_nodev/text dump/raw/train_nodev/text_spk${i}
    cp dump/raw/train_dev/text dump/raw/train_dev/text_spk${i}
    cp dump/raw/test/text dump/raw/test/text_spk${i}
    cp dump/raw/test_seg/text dump/raw/test_seg/text_spk${i}
done
./run_multispkr.sh --ngpu 0 --stage 10 --stop-stage 13 --skip-upload false --feats-type "raw" --token-type "bpe" \
    --feats_normalize "utterance_mvn" --asr-args "--max_epoch=1" --lm-args "--max_epoch=1" --python "${python}" \
    --asr_tag "train_multispkr_raw_en_bpe30" \
    --asr-args "--model_conf extract_feats_in_collect_stats=false --max_epoch=1 \
        --ctc_conf reduce=False --encoder transformer_multispkr \
        --encoder_conf num_blocks=2 --encoder_conf num_blocks_sd=2 --encoder_conf num_inf=2 \
        --decoder rnn \
        --model pit_espnet --model_conf num_inf=2 --model_conf num_ref=2 \
        --preprocessor multi --preprocessor_conf text_name='['text', 'text_spk2']'" \
    --inference-args "--multi_asr true"

# Remove generated files in order to reduce the disk usage
rm -rf exp dump data
cd "${cwd}"

# [ESPnet2] test tts recipe
cd ./egs2/mini_an4/tts1
gen_dummy_coverage
echo "==== [ESPnet2] TTS ==="
./run.sh --ngpu 0 --stage 1 --stop-stage 8 --skip-upload false  --train-args "--max_epoch 1" --python "${python}"
# Remove generated files in order to reduce the disk usage
rm -rf exp dump data

# [ESPnet2] test gan-tts recipe
# NOTE(kan-bayashi): pytorch 1.4 - 1.6 works but 1.6 has a problem with CPU,
#   so we test this recipe using only pytorch > 1.6 here.
#   See also: https://github.com/pytorch/pytorch/issues/42446
if python3 -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) > L("1.6")' &> /dev/null; then
    ./run.sh --fs 22050 --tts_task gan_tts --feats_extract linear_spectrogram --feats_normalize none --inference_model latest.pth \
        --ngpu 0 --stop-stage 8 --skip-upload false --train-args "--num_iters_per_epoch 1 --max_epoch 1" --python "${python}"
    rm -rf exp dump data
fi
cd "${cwd}"

# [ESPnet2] test enh recipe
if python -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.2.0")' &> /dev/null;  then
    cd ./egs2/mini_an4/enh1
    gen_dummy_coverage
    echo "==== [ESPnet2] ENH ==="
    ./run.sh --stage 1 --stop-stage 1 --python "${python}"
    feats_types="raw"
    for t in ${feats_types}; do
        echo "==== feats_type=${t} ==="
        ./run.sh --ngpu 0 --stage 2 --stop-stage 10 --skip-upload false --feats-type "${t}" --ref-num 1 --enh-args "--max_epoch=1" --python "${python}"
        ./run.sh --ngpu 0 --stage 2 --stop-stage 10 --skip-upload false --feats-type "${t}" --ref-num 1 --enh-args "--max_epoch=1" --python "${python}" --extra_wav_list "rirs.scp noises.scp" --enh_config ./conf/train_with_preprocessor.yaml
        ./run.sh --ngpu 0 --stage 2 --stop-stage 10 --skip-upload false --feats-type "${t}" --ref-num 1 --enh-args "--max_epoch=1" --python "${python}" --enh_config conf/train_with_dynamic_mixing.yaml --ref-num 2
    done
    # Remove generated files in order to reduce the disk usage
    rm -rf exp dump data
    cd "${cwd}"
fi

# [ESPnet2] test enh_tse recipe
if python -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.2.0")' &> /dev/null;  then
    cd ./egs2/mini_an4/tse1
    gen_dummy_coverage
    echo "==== [ESPnet2] ENH_TSE ==="
    feats_types="raw"
    for t in ${feats_types}; do
        echo "==== feats_type=${t} ==="
        ./run.sh --ngpu 0 --stage 1 --stop-stage 10 --skip-upload false --feats-type "${t}" --ref-num 1 --enh-args "--max_epoch=1" --python "${python}"
        ./run.sh --ngpu 0 --stage 1 --stop-stage 10 --skip-upload false --feats-type "${t}" --ref-num 1 --enh-args "--max_epoch=1" --python "${python}" --local_data_opts "--random-enrollment true" --enh_config ./conf/train_random_enrollment.yaml
    done
    # Remove generated files in order to reduce the disk usage
    rm -rf exp dump data
    cd "${cwd}"
fi

# [ESPnet2] test ssl1 recipe
if python3 -c 'import fairseq; import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.12.0")' &> /dev/null; then
    cd ./egs2/mini_an4/ssl1
    gen_dummy_coverage
    echo "==== [ESPnet2] SSL1/HUBERT ==="
    ./run.sh --ngpu 0 --stage 1 --stop-stage 7 --feats-type "raw" --token_type "word" --skip_upload_hf false \
        --hubert-args "--max_epoch=1" --python "${python}"
    # Remove generated files in order to reduce the disk usage
    rm -rf exp dump data
    cd "${cwd}"
fi

# [ESPnet2] test enh_asr1 recipe
if python -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.2.0")' &> /dev/null;  then
    cd ./egs2/mini_an4/enh_asr1
    gen_dummy_coverage
    echo "==== [ESPnet2] ENH_ASR ==="
    ./run.sh --ngpu 0 --stage 0 --stop-stage 15 --skip-upload_hf false --feats-type "raw" --spk-num 1 --enh_asr_args "--max_epoch=1 --enh_separator_conf num_spk=1 --asr_decoder rnn" --python "${python}"
    # Remove generated files in order to reduce the disk usage
    rm -rf exp dump data
    cd "${cwd}"
fi

# [ESPnet2] test st recipe
cd ./egs2/mini_an4/st1
echo "==== [ESPnet2] ST ==="
./run.sh --stage 1 --stop-stage 1
feats_types="raw fbank_pitch"
token_types="bpe char"
for t in ${feats_types}; do
    ./run.sh --stage 2 --stop-stage 4 --feats-type "${t}" --python "${python}"
done
for t in ${token_types}; do
    ./run.sh --stage 5 --stop-stage 5 --tgt_token_type "${t}" --src_token_type "${t}" --python "${python}"
done
for t in ${feats_types}; do
    for t2 in ${token_types}; do
        echo "==== feats_type=${t}, token_types=${t2} ==="
        ./run.sh --ngpu 0 --stage 6 --stop-stage 13 --skip-upload false --feats-type "${t}" --tgt_token_type "${t2}" --src_token_type "${t2}" \
            --st-args "--max_epoch=1" --lm-args "--max_epoch=1" --inference_args "--beam_size 5" --python "${python}"
    done
done
echo "==== feats_type=raw, token_types=bpe, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn ==="
./run.sh --ngpu 0 --stage 10 --stop-stage 13 --skip-upload false --feats-type "raw" --tgt_token_type "bpe" --src_token_type "bpe" \
    --feats_normalize "utterance_mvn" --lm-args "--max_epoch=1" --inference_args "--beam_size 5" --python "${python}" \
    --st-args "--model_conf extract_feats_in_collect_stats=false --max_epoch=1"

echo "==== use_streaming, feats_type=raw, token_types=bpe, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn ==="
./run.sh --use_streaming true --ngpu 0 --stage 6 --stop-stage 13 --skip-upload false --feats-type "raw" --tgt_token_type "bpe" --src_token_type "bpe" \
    --feats_normalize "utterance_mvn" --lm-args "--max_epoch=1" --inference_args "--beam_size 5" --python "${python}" \
    --st-args "--model_conf extract_feats_in_collect_stats=false --max_epoch=1 --encoder=contextual_block_transformer --decoder=transformer
                --encoder_conf block_size=40 --encoder_conf hop_size=16 --encoder_conf look_ahead=16"

# Remove generated files in order to reduce the disk usage
rm -rf exp dump data
cd "${cwd}"

# [ESPnet2] Validate configuration files
echo "<blank>" > dummy_token_list
echo "==== [ESPnet2] Validation configuration files ==="
if python3 -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.8.0")' &> /dev/null;  then
    for f in egs2/*/asr1/conf/train_asr*.yaml; do
        if [ "$f" == "egs2/fsc/asr1/conf/train_asr.yaml" ]; then
            if ! python3 -c "import s3prl" > /dev/null; then
                continue
            fi
        fi
        if [ "$f" == "egs2/how2_2000h/asr1/conf/train_asr_conformer_lf.yaml" ]; then
            if ! python3 -c "import longformer" > /dev/null; then
                continue
            fi
        fi
        ${python} -m espnet2.bin.asr_train --config "${f}" --iterator_type none --dry_run true --output_dir out --token_list dummy_token_list
    done

    for f in egs2/*/asr1/conf/train_lm*.yaml; do
        ${python} -m espnet2.bin.lm_train --config "${f}" --iterator_type none --dry_run true --output_dir out --token_list dummy_token_list
    done

    for f in egs2/*/tts1/conf/train*.yaml; do
        ${python} -m espnet2.bin.tts_train --config "${f}" --iterator_type none --normalize none --dry_run true --output_dir out --token_list dummy_token_list
    done

    for f in egs2/*/enh1/conf/train*.yaml; do
        ${python} -m espnet2.bin.enh_train --config "${f}" --iterator_type none --dry_run true --output_dir out
    done

    if python3 -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.12.0")' &> /dev/null; then
        for f in egs2/*/ssl1/conf/train*.yaml; do
            ${python} -m espnet2.bin.hubert_train --config "${f}" --iterator_type none --normalize none --dry_run true --output_dir out --token_list dummy_token_list --num_classes 10
        done
    fi

    for f in egs2/*/enh_asr1/conf/train_enh_asr*.yaml; do
        ${python} -m espnet2.bin.enh_s2t_train --config "${f}" --iterator_type none --dry_run true --output_dir out --token_list dummy_token_list
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

coverage combine egs2/*/*/.coverage
coverage report
coverage xml
