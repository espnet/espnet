
run_as2m() {
    local ckpt=$1
    local run_name=$2
    config=ear_large
    if [[ $run_name == *"earbase"* ]]; then
        config=ear_base
    fi
    ./run.sh --config_prefix $config \
             --run_name ${run_name} \
             --template_args CHECKPOINT_PATH:${ckpt} \
             --parallel true \
             --recipe audioset2m \
             --log_wandb true \
             --task_args "--wandb_entity shikhar" \
             --recipe_args "--stage 6 --ngpu 5"
}

run_as20k() {
    local ckpt=$1
    local run_name=$2
    config=ear_large
    if [[ $run_name == *"earbase"* ]]; then
        config=ear_base
    fi
    ./run.sh --config_prefix $config \
             --run_name ${run_name} \
             --template_args CHECKPOINT_PATH:${ckpt} \
             --parallel true \
             --recipe audioset20k \
             --log_wandb true \
             --task_args "--wandb_entity shikhar" \
             --recipe_args "--stage 6 --ngpu 1"
}


run_rfcx() {
    local ckpt=$1
    local run_name=$2
    config=ear_large
    if [[ $run_name == *"earbase"* ]]; then
        config=ear_base
    fi
    ./run.sh --config_prefix $config \
             --run_name ${run_name} \
             --template_args CHECKPOINT_PATH:${ckpt} \
             --parallel true \
             --recipe beans_rfcx \
             --log_wandb true \
             --task_args "--wandb_entity shikhar" \
             --recipe_args "--stage 6"
}

run_aqa_open_clap() {
    local ckpt=$1
    local run_name=$2
    config=ear_large
    if [[ $run_name == *"earbase"* ]]; then
        config=ear_base
    fi
    ./run.sh --config_prefix $config \
             --run_name ${run_name} \
             --template_args CHECKPOINT_PATH:${ckpt} \
             --parallel true \
             --recipe aqa_open_clap \
             --log_wandb true \
             --task_args "--wandb_entity shikhar" \
             --recipe_args "--stage 6"
}

run_rfcx_dasheng() {
    local ckpt=$1
    local run_name=$2
    ./run.sh --config_prefix $run_name \
            --run_name $run_name \
            --template_args DASHENG_MODEL_NAME:$ckpt \
            --parallel true \
            --recipe beans_rfcx \
            --log_wandb true \
            --task_args "--wandb_entity shikhar" \
            --recipe_args "--stage 6"
}

# run_rfcx_dasheng mispeech/dasheng-1.2B dasheng_1.2b &
# run_rfcx_dasheng mispeech/dasheng-0.6B dasheng_600m &
# run_rfcx_dasheng mispeech/dasheng-base dasheng_base &


# AUDIOSET-2M
run_as2m /work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_base/beats_iter0_base.tune_lr5e-4_warmup40000_bins1600000_totalsteps400000/epoch59.pt earbasei1 &
run_as2m /work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_base/beats_iter1_base.tune_lr5e-4_warmup40000_bins1600000_totalsteps400000/epoch59.pt earbasei2 &
run_as2m /work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter2_base2.tune_lr5e-4_warmup40000_bins1600000_totalsteps400000/epoch_latest.pt earbasei3 &

# run_as2m /work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_large/beats_iter0_large.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000/epoch59.pt earlarge1 &
# run_as2m /work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter1_large1.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000/epoch_latest.pt earlarge2 &
# run_as2m /work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter2_large2.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000/epoch_latest.pt earlarge3 &


# AUDIOSET-20K
run_as20k /work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_base/beats_iter0_base.tune_lr5e-4_warmup40000_bins1600000_totalsteps400000/epoch59.pt earbasei1 &
run_as20k /work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_base/beats_iter1_base.tune_lr5e-4_warmup40000_bins1600000_totalsteps400000/epoch59.pt earbasei2 &
run_as20k /work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter2_base2.tune_lr5e-4_warmup40000_bins1600000_totalsteps400000/epoch_latest.pt earbasei3 &

# run_as20k /work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_large/beats_iter0_large.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000/epoch59.pt earlarge1 &
# run_as20k /work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter1_large1.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000/epoch_latest.pt earlarge2 &
# run_as20k /work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter2_large2.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000/epoch_latest.pt earlarge3 &


# run_rfcx /work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_base/beats_iter0_base.tune_lr5e-4_warmup40000_bins1600000_totalsteps400000/epoch59.pt earbasei1 &
# run_rfcx /work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_base/beats_iter1_base.tune_lr5e-4_warmup40000_bins1600000_totalsteps400000/epoch59.pt earbasei2 &
# run_rfcx /work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter2_base2.tune_lr5e-4_warmup40000_bins1600000_totalsteps400000/epoch_latest.pt earbasei3 &
# run_rfcx /work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_large/beats_iter0_large.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000/epoch59.pt earlarge1 &
# run_rfcx /work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter1_large1.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000/epoch_latest.pt earlarge2 &
# run_rfcx /work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter2_large2.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000/epoch_latest.pt earlarge3 &

# run_aqa_open_clap /work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter2_base2.tune_lr5e-4_warmup40000_bins1600000_totalsteps400000/epoch_latest.pt earbasei3 &
# run_aqa_open_clap /work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter1_large1.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000/epoch_latest.pt earlarge2 &
# run_aqa_open_clap /work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter2_large2.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000/epoch_latest.pt earlarge3 &

wait
