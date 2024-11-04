tag=ssl_asr_tedlium2_train_delay_asr_small
python ../../../utils/average_checkpoints.py \
    --snapshots exp/speechlm_${tag}/checkpoint_{2,3,4,5,6,7}.pth \
    --out exp/speechlm_${tag}/2_7epoch.pth \
    --backend pytorch \
    --num 6