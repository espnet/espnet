# #!/usr/bin/env bash
# # Set bash to 'debug' mode, it will exit on :
# # -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# set -e
# set -u
# set -o pipefail

# # 1-4 : cpu (data prep: local, format, filter, fbank)
# # 5: gpu (tokenization)
# # 6: cpu (collect stats)
# # 7: gpu (training)

# . ./db.sh

# timestamp=$(date "+%m%d.%H%M%S")
# mynametag=

# ssl_tag=${mynametag}.${timestamp}

# ngpu=2
# num_splits_ssl=1

# storage_dir=/work/nvme/bbjs/sbharadwaj/watkins_ssl
# mkdir -p "${storage_dir}"

# ######
# tokenizer_train_config=
# external_teacher_model=

# external_tokenizer_model=/work/nvme/bbjs/sbharadwaj/BEATs_models/Tokenizer_iter3.pt
# ######
# # master_tapes


# for train_set in cut_tapes cut_tapes_noise; do
#     for iter in 0 1;do
#         train_start_iter=${iter}
#         train_stop_iter=${iter}
#         tokenizer_inf_config=conf/beats_watkins_tokinf_config_${train_set}.yaml
#         train_config=conf/beats_watkins_train_config_${train_set}.yaml
#         ssl_tag=10k_ckpt.${train_set}.${iter}.100k_total
#         use_wandb=true
#         wandb_project=BEATsPT_Watkins
#         ./beats.sh \
#             --speech_fold_length 160000 \
#             --text_fold_length 600 \
#             --ssl_tag ${ssl_tag} \
#             --n_targets 1024 \
#             --datadir "${storage_dir}/data" \
#             --dumpdir "${storage_dir}/dump" \
#             --expdir "${storage_dir}/exp_${train_set}" \
#             --stage 7 \
#             --stop_stage 7 \
#             --feats_type fbank \
#             --ngpu ${ngpu} \
#             --num_nodes 1 \
#             --train_start_iter "${train_start_iter}"\
#             --train_stop_iter "${train_stop_iter}" \
#             --nj 32 \
#             --max_wav_duration 11 \
#             --min_wav_duration 0.3 \
#             --external_teacher_model "${external_teacher_model}" \
#             --external_tokenizer_model "${external_tokenizer_model}" \
#             --tokenizer_train_config "${tokenizer_train_config}" \
#             --tokenizer_inference_config "${tokenizer_inf_config}" \
#             --tokenizer_inference_batch_size 160 \
#             --train_config "${train_config}" \
#             --train_set "${train_set}" \
#             --valid_set "${train_set}" \
#             --num_splits_ssl ${num_splits_ssl} \
#             --beats_args "--use_wandb ${use_wandb} --wandb_project ${wandb_project} --wandb_name ${ssl_tag} --wandb_entity shikhar" &
#     done
# done

# wait


# CONVERT CKPT
# for set in cut_tapes cut_tapes_noise; do
#     for iter in 0 1; do
#         python /work/nvme/bbjs/sbharadwaj/espnet/espnet2/beats/generate_beats_checkpoint.py \
#              --espnet_model_checkpoint_path /work/nvme/bbjs/sbharadwaj/watkins_ssl/exp_${set}/beats_iter${iter}_10k_ckpt.${set}.${iter}.100k_total/checkpoint_10/10/mp_rank_00_model_states.pt \
#              --output_path /work/nvme/bbjs/sbharadwaj/watkins_ssl/exp_${set}/beats_iter${iter}_10k_ckpt.${set}.${iter}.100k_total/epoch10.pt \
#              --espnet_model_config_path /work/nvme/bbjs/sbharadwaj/watkins_ssl/exp_${set}/beats_iter${iter}_10k_ckpt.${set}.${iter}.100k_total/config.yaml \
#              --deepspeed_checkpoint
#     done
# done


# RUN WATKINS EVAL
for set in cut_tapes cut_tapes_noise; do
    for iter in 0 1; do
        ckpt_path=/work/nvme/bbjs/sbharadwaj/watkins_ssl/exp_${set}/beats_iter${iter}_10k_ckpt.${set}.${iter}.100k_total/epoch10.pt
        run_config=/work/nvme/bbjs/sbharadwaj/espnet/egs2/watkins/ssl1/conf/eval_config.freeze.${set}.${iter}.yaml
        sed "s|CHECKPOINT_PATH|${ckpt_path}|g" conf/eval_config.yaml > ${run_config}
        storage_dir_=/work/nvme/bbjs/sbharadwaj/watkins_ssl
        datadir=/work/nvme/bbjs/sbharadwaj/espnet/egs2/audioverse/v1/data/beans_watkins
        dumpdir=/work/nvme/bbjs/sbharadwaj/espnet/egs2/audioverse/v1/dump/beans_watkins
        expdir=${storage_dir_}/exp/${set}.${iter}
        (cd ../../beans/cls1 && ./run_watkins.sh --cls_tag freeze.${set}.${iter} --datadir ${datadir} --dumpdir ${dumpdir} --expdir ${expdir} --stage 6 --cls_config ${run_config}) &
        echo "----"
    done
done

wait