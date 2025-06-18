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
             --recipe_args "--stage 6 --ngpu 4"
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

run_as20k_beats() {
    local ckpt=$1
    local run_name=$2
    ./run.sh --config_prefix beats \
             --run_name ${run_name} \
             --template_args CHECKPOINT_PATH:${ckpt} \
             --parallel true \
             --recipe audioset20k \
             --log_wandb true \
             --task_args "--wandb_entity shikhar" \
             --recipe_args "--stage 6 --ngpu 1"
}

# AUDIOSET-2M
# run_as2m /work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_base/beats_iter0_base.tune_lr5e-4_warmup40000_bins1600000_totalsteps400000/epoch59.pt earbasei1.fix2x &
# run_as2m /work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_base/beats_iter1_base.tune_lr5e-4_warmup40000_bins1600000_totalsteps400000/epoch59.pt earbasei2.fix2x &
# run_as2m /work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter2_base2.tune_lr5e-4_warmup40000_bins1600000_totalsteps400000/epoch_latest.pt earbasei3.fix2x &

run_as2m /work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_large/beats_iter0_large.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000/epoch59.pt earlarge1.final &
run_as2m /work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter1_large1.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000/epoch_latest.pt earlarge2.final &
run_as2m /work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter2_large2.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000/epoch_latest.pt earlarge3.final &

# AUDIOSET-20K
# run_as20k /work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_base/beats_iter0_base.tune_lr5e-4_warmup40000_bins1600000_totalsteps400000/epoch59.pt earbasei1.fix2x &
# run_as20k /work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_base/beats_iter1_base.tune_lr5e-4_warmup40000_bins1600000_totalsteps400000/epoch59.pt earbasei2.fix2x &
# run_as20k /work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter2_base2.tune_lr5e-4_warmup40000_bins1600000_totalsteps400000/epoch_latest.pt earbasei3.fix4x.lre5 &

# run_as20k /work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_large/beats_iter0_large.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000/epoch59.pt earlarge1 &
# run_as20k /work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter1_large1.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000/epoch_latest.pt earlarge2 &
# run_as20k /work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter2_large2.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000/epoch_latest.pt earlarge3 &

# run_as20k_beats /work/nvme/bbjs/sbharadwaj/model_checkpoints/BEATs_iter3.pt beatsi3.epoch2x.lr3e5 &

wait