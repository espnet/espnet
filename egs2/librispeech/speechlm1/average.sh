tag=librispeech_960
python ../../../utils/average_checkpoints.py \
    --snapshots exp/speechlm_${tag}/checkpoint_{7,8,9,10,11,12,13,14,15,16}.pth \
    --out exp/speechlm_${tag}/7_16epoch.pth \
    --backend pytorch \
    --num 10