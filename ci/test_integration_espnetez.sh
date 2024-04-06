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
cd ${cwd}/egs2/mini_an4/lm1 || exit
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
cd ${cwd}/egs2/mini_an4/s2t1 || exit
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
cd ${cwd}/egs2/mini_an4/tts1 || exit
rm -rf exp data dump

echo "==== [ESPnet2] TTS ==="
# data preparation
./run.sh --ngpu 0 --stage 1 --stop-stage 4 --skip-upload false --python "${python}" --train-args "--num_workers 0"
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

Remove generated files in order to reduce the disk usage
rm -rf exp data dump


# [ESPnet Easy] test gan-tts recipe with coverage
# ./run.sh --fs 22050 --tts_task gan_tts --feats_extract linear_spectrogram --feats_normalize none --inference_model latest.pth \
#         --ngpu 0 --stop-stage 4

# python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
#     --task tts \
#     --data_path data \
#     --train_dump_path dump/raw/train_nodev \
#     --valid_dump_path dump/raw/train_dev \
#     --exp_path ./exp \
#     --config_path conf/train_vits_debug.yaml \
#     --train_sentencepiece_model \
#     --run_collect_stats \
#     --run_train

# # finetune
# python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
#     --task tts \
#     --data_path data \
#     --train_dump_path dump/raw/train_nodev \
#     --valid_dump_path dump/raw/train_dev \
#     --exp_path ./exp \
#     --config_path conf/train_vits_debug.yaml \
#     --run_finetune

# # Remove generated files in order to reduce the disk usage
# rm -rf exp data dump


# [ESPnet Easy] test asr2 recipe with coverage
cd ${cwd}/egs2/mini_an4/asr2 || exit
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
cd "${cwd}" || exit


# [ESPnet Easy] test enh recipe with coverage
if python -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.2.0")' &> /dev/null; then
    cd ${cwd}/egs2/mini_an4/enh1 || exit
    rm -rf exp data dump
    gen_dummy_coverage
    echo "==== [ESPnet2] ENH ==="
    ./run.sh --stage 1 --stop-stage 4 --python "${python}" --extra_wav_list "rirs.scp noises.scp"

    configs=("train_with_preprocessor_debug" "train_with_data_aug_debug" "train_debug")
    for conf in "${configs[@]}"; do
        rm -rf exp data/spm
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

    cd "${cwd}" || exit
fi

# [ESPnet Easy] test enh-tse recipe with coverage
if python -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.2.0")' &> /dev/null; then
    cd ${cwd}/egs2/mini_an4/tse1 || exit
    rm -rf exp data dump
    gen_dummy_coverage
    echo "==== [ESPnet2] ENH_TSE ==="

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

    # finetune
    python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
        --task enh_tse \
        --data_path data \
        --train_dump_path dump/raw/train_nodev \
        --valid_dump_path dump/raw/train_dev \
        --exp_path ./exp \
        --config_path ./conf/train_debug.yaml \
        --run_finetune

    # ./run.sh --ngpu 0 --stage 3 --stop-stage 4 --skip-upload false --feats-type "${t}" --ref-num 1 --python "${python}" \
    #         --train_set train_nodev_unk_nspk --valid_set test_unk_nspk --test_sets "train_dev_unk_nspk" \
    #         --enh_config ./conf/train_variable_nspk_debug.yaml --enh-args "--num_workers 0" --variable_num_refs true

    # python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    #     --task enh_tse \
    #     --data_path data \
    #     --train_dump_path dump/raw/train_nodev_unk_nspk \
    #     --valid_dump_path dump/raw/test_unk_nspk \
    #     --exp_path ./exp \
    #     --config_path ./conf/train_variable_nspk_debug.yaml \
    #     --variable_num_refs \
    #     --run_collect_stats \
    #     --run_train

    # rm -rf exp dump data
    # ./run.sh --ngpu 0 --stage 1 --stop-stage 3 --feats-type "raw" --ref-num 1 \
    #         --local_data_opts "--random-enrollment true" \
    #         --enh_config ./conf/train_random_enrollment_debug.yaml --enh-args "--num_workers 0"

    # python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    #     --task enh_tse \
    #     --data_path data \
    #     --train_dump_path dump/raw/org/train_nodev \
    #     --valid_dump_path dump/raw/train_dev \
    #     --exp_path ./exp \
    #     --config_path ./conf/train_random_enrollment_debug.yaml \
    #     --run_collect_stats \
    #     --run_train

    # ./run.sh --ngpu 0 --stage 3 --stop-stage 6 --feats-type "raw" --ref-num 1 \
    #         --train_set train_nodev_unk_nspk --valid_set test_unk_nspk --test_sets "train_dev_unk_nspk" \
    #         --enh_config ./conf/train_variable_nspk_random_enrollment_debug.yaml --enh-args "--num_workers 0" --variable_num_refs true

    # python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    #     --task enh_tse \
    #     --data_path data \
    #     --train_dump_path dump/raw/org/train_nodev_unk_nspk \
    #     --valid_dump_path dump/raw/test_unk_nspk \
    #     --exp_path ./exp \
    #     --config_path ./conf/train_variable_nspk_random_enrollment_debug.yaml \
    #     --variable_num_refs \
    #     --run_collect_stats \
    #     --run_train

fi

# [ESPnet Easy] test enh-asr recipe with coverage
if python -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.2.0")' &> /dev/null; then
    cd ${cwd}/egs2/mini_an4/enh_asr1 || exit
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
    cd "${cwd}" || exit
fi

# [ESPnet Easy] test ssl recipe with coverage
if python3 -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.12.0")' &> /dev/null; then
    cd ${cwd}/egs2/mini_an4/ssl1 || exit
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
    cd "${cwd}" || exit
fi

# [ESPnet Easy] test st recipe with coverage
cd ${cwd}/egs2/mini_an4/st1 || exit
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
rm -rf exp data/spm
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
cd ${cwd}/egs2/mini_an4/s2t1 || exit
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


# [ESPnet Easy] test s2st1 recipe with coverage
cd ${cwd}/egs2/mini_an4/s2st1 || exit
rm -rf exp dump data
gen_dummy_coverage
echo "==== [ESPnet2] S2ST ==="
./run.sh --ngpu 0 --stage 1 --stop_stage 5 --use_discrete_unit false --s2st_config conf/s2st_spec_debug.yaml
python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    --task s2st \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path ./conf/s2st_spec_debug.yaml \
    --run_collect_stats \
    --run_train

python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
    --task s2st \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path ./conf/s2st_spec_debug.yaml \
    --run_finetune

rm -rf exp dump data ckpt

./run.sh --ngpu 0 --stage 1 --stop_stage 5 --python "${python}" --use_discrete_unit true \
    --s2st_config conf/train_s2st_discrete_unit_debug.yaml --clustering_num_threads 2 --feature_num_clusters 5

python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
    --task s2st \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path ./conf/train_s2st_discrete_unit_debug.yaml \
    --run_collect_stats \
    --run_train \
    --use_discrete_unit

python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
    --task s2st \
    --data_path data \
    --train_dump_path dump/raw/train_nodev \
    --valid_dump_path dump/raw/train_dev \
    --exp_path ./exp \
    --config_path ./conf/train_s2st_discrete_unit_debug.yaml \
    --run_finetune \
    --use_discrete_unit


# Remove generated files in order to reduce the disk usage
rm -rf exp dump data ckpt .cache

# [ESPnet Easy] test spk recipe with coverage
cd ${cwd}/egs2/mini_an4/spk1 || exit
rm -rf exp dump data
gen_dummy_coverage
echo "==== [ESPnet2] SPK1 ==="

# data preparation
./run.sh --ngpu 0 --stage 0 --stop-stage 3 --feats-type "raw" --spk-args "--num_workers 0"

spk_configs=("train_rawnet3_dataaug_debug" "train_rawnet3_sampler" "train_ecapa" \
"train_xvector" "train_ska" "train_identity" "train_conformer" "train_rawnet3_sampler")

for conf in "${spk_configs[@]}"; do
    rm -rf exp
    python -m coverage run --append ../../../test/espnetez/test_integration_espnetez.py \
        --task spk \
        --data_path data \
        --train_dump_path dump/raw/train_nodev_sp \
        --valid_dump_path dump/raw/train_dev \
        --exp_path ./exp \
        --config_path ./conf/train_rawnet3_dataaug_debug.yaml \
        --run_collect_stats \
        --run_train

    python -m coverage run --append ../../../test/espnetez/test_integration_espnetez_ft.py \
        --task spk \
        --data_path data \
        --train_dump_path dump/raw/train_nodev_sp \
        --valid_dump_path dump/raw/train_dev \
        --exp_path ./exp \
        --config_path ./conf/train_rawnet3_dataaug_debug.yaml \
        --run_finetune

    done

cd "${cwd}" || exit


echo "=== report ==="
python -m coverage combine egs2/*/*/.coverage
python -m coverage report
python -m coverage xml
