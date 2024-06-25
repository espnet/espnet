#!/bin/bash

path=/ocean/projects/cis210027p/jiatong/codec/espnet/egs2/amuse/codec1

conf=conf/train_soundstream4_fs44100.yaml
tag=v4_fs44100

sid=$(
    sbatch -p RM-shared --time 2-0:00:00 \
    --output=${path}/cycle1_normal_${tag}.out \
    --error=${path}/cycle1_normal_${tag}.err \
    ./run.sh --stage 5 --stop_stage 5 --train_config ${conf} | awk '{print $NF}'
)

for i in  2 3 4 5 6 7 8; do

sid=$(
    sbatch -p RM-shared --time 2-0:00:00 \
    --output=${path}/cycle${i}_normal_${tag}.out \
    --error=${path}/cycle${i}_normal_${tag}.err \
    --dependency=afterany:${sid} \
    ./run.sh --stage 5 --stop_stage 5 --train_config ${conf} | awk '{print $NF}'
)

done

# sbatch --dependency=afterany:${sid} --time 2-0:00:00 -p RM-shared -N 1 \
#        --output=${path}/cycle${i}_normal_${tag}_decode.out \
#        --error=${path}/cycle${i}_normal_${tag}_decode.err \
#        --dependency=afterany:${sid} \
#        ./run.sh --stage 6 --stop_stage 6 --train_config ${conf} | awk '{print $NF}'

# sbatch --dependency=afterany:${sid} --time 2-0:00:00 -p RM-shared -N 1 --cpus-per-task=16 --mem=32000M ./run.sh --stage 7 --train_config ${conf}
