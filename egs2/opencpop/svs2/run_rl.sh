#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# spectrogram-related arguments
fs=16000
fmin=80
fmax=7600
n_fft=2048
n_shift=320
win_length=1200
score_feats_extract=syllable_score_feats   # frame_score_feats | syllable_score_feats

# discrete related
kmeans_feature="multi/hubert_large_6+wavlm_large_6+wavlm_large_23" # split with '/', e.g. "multi/wavlm6+large6", "wavlm_large/6" | "encodec/1" | "xls_r_300m/6", use model_type/layer_index
multi_token="hubert_large_ll60k_128_6_RVQ_0 wavlm_large_128_6_RVQ_0 wavlm_large_128_23_RVQ_0" # split with ' '
mix_type="frame" # frame | sequencee
nclusters=128
RVQ_layers=2
preset_layer=none
preset_token=none

# set
train_set=tr_no_dev
valid_set=dev
test_sets="dev test"
select_sets="${valid_set} ${train_set}"

# config
train_config=conf/tuning/train_naive_rnn_dp.yaml
inference_config=conf/tuning/decode.yaml

# preprocessing arguments
g2p=none
cleaner=none
pitch_extract=dio

# infer
gpu_inference=true

# rl related
prep_rl_data=false
sample_data=false
samples_num=10
rl_data="dump/rl"
select_metrics="spk_similarity"
use_refsvs=false

pretrain_checkpoint="exp/svs_train_toksing_raw_phn_none_zh/valid.loss.best.pth"

versa_path="/data7/tyx/versa"

stage=1
stop_stage=2

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Prepare dataset for training"
    
    sub_stage=1
    sub_stop_stage=4
    train_config=config/tuning/train_toksing.yaml

    if [ ${sub_stage} -le 1 ] && [ ${sub_stop_stage} -ge 1 ]; then
        log "substage 1.1: get sample idx"
        prep_rl_data=true
        sample_data=true

        ./svs2.sh \
            --lang zh \
            --stage 8 \
            --stop_stage 8 \
            --local_data_opts "--stage 0" \
            --feats_type raw \
            --pitch_extract "${pitch_extract}" \
            --fs "${fs}" \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft "${n_fft}" \
            --n_shift "${n_shift}" \
            --win_length "${win_length}" \
            --token_type phn \
            --g2p ${g2p} \
            --cleaner ${cleaner} \
            --preset_layer ${preset_layer} \
            --preset_token ${preset_token} \
            --train_config "${train_config}" \
            --inference_config "${inference_config}" \
            --gpu_inference ${gpu_inference} \
            --train_set "${train_set}" \
            --valid_set "${valid_set}" \
            --test_sets "${select_sets}" \
            --score_feats_extract "${score_feats_extract}" \
            --srctexts "data/${train_set}/text" \
            --RVQ_layers "${RVQ_layers}" \
            --kmeans_opts "--batch_bins 4800000" \
            --kmeans_feature "${kmeans_feature}" \
            --multi_token "${multi_token}" \
            --mix_type "${mix_type}" \
            --nclusters "${nclusters}" \
            --RVQ_layers "${RVQ_layers}" \
            --ngpu 1 \
            --prep_rl_data "${prep_rl_data}" \
            --sample_data "${sample_data}" \
            --samples_num "${samples_num}" \
            "$@"
    fi

    if [ ${sub_stage} -le 2 ] && [ ${sub_stop_stage} -ge 2 ]; then
        log "substage 1.2: generate wav for corresponding idx"
        prep_rl_data=true
        sample_data=false

        ./svs2.sh \
            --lang zh \
            --stage 8 \
            --stop_stage 8 \
            --local_data_opts "--stage 0" \
            --feats_type raw \
            --pitch_extract "${pitch_extract}" \
            --fs "${fs}" \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft "${n_fft}" \
            --n_shift "${n_shift}" \
            --win_length "${win_length}" \
            --token_type phn \
            --g2p ${g2p} \
            --cleaner ${cleaner} \
            --preset_layer ${preset_layer} \
            --preset_token ${preset_token} \
            --train_config "${train_config}" \
            --inference_config "${inference_config}" \
            --gpu_inference ${gpu_inference} \
            --train_set "${train_set}" \
            --valid_set "${valid_set}" \
            --test_sets "${select_sets}" \
            --score_feats_extract "${score_feats_extract}" \
            --srctexts "data/${train_set}/text" \
            --RVQ_layers "${RVQ_layers}" \
            --kmeans_opts "--batch_bins 4800000" \
            --kmeans_feature "${kmeans_feature}" \
            --multi_token "${multi_token}" \
            --mix_type "${mix_type}" \
            --nclusters "${nclusters}" \
            --RVQ_layers "${RVQ_layers}" \
            --ngpu 1 \
            --prep_rl_data "${prep_rl_data}" \
            --sample_data "${sample_data}" \
            --samples_num "${samples_num}" \
            "$@"

        for dset in ${select_sets}; do
            # preprocess wav path
            org_path="data/${dset}/wav.scp"
            pred_path="${rl_data}/${dset}/samples/wav.scp"
            gt_path="${rl_data}/${dset}/samples/wav_gt.scp"
            awk -v samples_num=$samples_num '
            {
                uid = $1
                wav_path = $2
                cmd = "realpath " wav_path
                cmd | getline absolute_path
                close(cmd)

                for (i = 0; i < samples_num; i++) {
                    print uid "_" i " " absolute_path
                }
            }
            ' ${org_path} > ${gt_path}
        done
    fi

    if [ ${sub_stage} -le 3 ] && [ ${sub_stop_stage} -ge 3 ]; then
        log "substage 1.3: annotate data with versa"
        if [ ! -e versa ]; then
            ln -s ${versa_path} versa
        fi
        for dset in ${select_sets}; do
            # Metrics
            src_dir="$(pwd)/${rl_data}/${dset}/samples"
            tgt_dir="$(pwd)/${rl_data}/${dset}/eval_metrics/raw"
            mkdir -p ${tgt_dir}
            for metric in ${select_metrics}; do
                echo ${tgt_dir}/eval_${metric}.txt
                if [ ${metric} == "mcd" ]; then
                    cd versa
                    python versa/bin/scorer.py \
                    --score_config egs/separate_metrics/mcd_f0.yaml \
                    --pred ${src_dir}/wav.scp \
                    --gt ${src_dir}/wav_gt.scp \
                    --output_file ${tgt_dir}/eval_${metric}.txt \
                    --use_gpu true
                    cd ..
                elif [ ${metric} == "spk_similarity" ]; then
                    cd versa
                    python versa/bin/scorer.py \
                    --score_config egs/separate_metrics/spk_similarity.yaml \
                    --pred ${src_dir}/wav.scp \
                    --gt ${src_dir}/wav_gt.scp \
                    --output_file ${tgt_dir}/eval_${metric}.txt \
                    --use_gpu true
                    cd ..
                fi
            done
        done
    fi
    
    if [ ${sub_stage} -le 4 ] && [ ${sub_stop_stage} -ge 4 ]; then
        log "substage 1.4: build dataset with annotated data"

        for dset in ${select_sets}; do
            # Metrics
            src_dir=${rl_data}/${dset}/eval_metrics/raw
            tgt_dir=${rl_data}/${dset}/
            sample_dir=${rl_data}/${dset}/samples
            metrics_opts=
            for metric in ${select_metrics}; do
                if [ ${metric} == "singmos" ]; then
                    metrics_opts+="--metric_names singmos "
                    metrics_opts+="--metric_weight 1.0 "
                    metrics_opts+="--metric_files ${src_dir}/eval_${metric}.txt "
                elif [ ${metric} == "mcd" ]; then
                    metrics_opts+="--metric_names mcd "
                    metrics_opts+="--metric_weight 1.0 "
                    metrics_opts+="--metric_files ${src_dir}/eval_${metric}.txt "
                elif [ ${metric} == "f0_rmse" ]; then
                    metrics_opts+="--metric_names f0_rmse "
                    metrics_opts+="--metric_weight 1.0 "
                    metrics_opts+="--metric_files ${src_dir}/eval_${metric}.txt "
                elif [ ${metric} == "spk_similarity" ]; then
                    metrics_opts+="--metric_names spk_similarity "
                    metrics_opts+="--metric_weight 1.0 "
                    metrics_opts+="--metric_files ${src_dir}/eval_${metric}.txt "
                fi
            done

            python pyscripts/utils/build_hfrl_dataset.py \
            --path_modality_types "${sample_dir}/samples_idx.scp,idx,npy" \
            --output_dir ${tgt_dir} \
            ${metrics_opts}
        done
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: train model with RL"
    
    prep_rl_data=false
    train_rl=true
    use_refsvs=true
    train_config=config/tuning/train_dpo.yaml
    if ${use_refsvs}; then
        rl_train_args+=" --init_param ${pretrain_checkpoint}:svs:svs ${pretrain_checkpoint}:svs:ref_svs "
    else
        rl_train_args+=" --init_param ${pretrain_checkpoint}:svs:svs "
    fi
    ./svs2.sh \
            --lang zh \
            --stage 7 \
            --stop_stage 7 \
            --local_data_opts "--stage 0" \
            --feats_type raw \
            --pitch_extract "${pitch_extract}" \
            --fs "${fs}" \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft "${n_fft}" \
            --n_shift "${n_shift}" \
            --win_length "${win_length}" \
            --token_type phn \
            --g2p ${g2p} \
            --cleaner ${cleaner} \
            --preset_layer ${preset_layer} \
            --preset_token ${preset_token} \
            --train_config "${train_config}" \
            --inference_config "${inference_config}" \
            --train_set "${train_set}" \
            --valid_set "${valid_set}" \
            --test_sets "${test_sets}" \
            --score_feats_extract "${score_feats_extract}" \
            --srctexts "data/${train_set}/text" \
            --RVQ_layers "${RVQ_layers}" \
            --kmeans_opts "--batch_bins 4800000" \
            --kmeans_feature "${kmeans_feature}" \
            --multi_token "${multi_token}" \
            --mix_type "${mix_type}" \
            --nclusters "${nclusters}" \
            --RVQ_layers "${RVQ_layers}" \
            --ngpu 1 \
            --train_rl "${train_rl}" \
            --train_args "${rl_train_args}" \
            "$@"

fi