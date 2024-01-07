#!/bin/bash
set -e
set -u
set -o pipefail
echo "################STAGE1###################"
bash data.sh
echo "################STAGE2###################"
./asr.sh --stage 2 --stop_stage 2 --speed_perturb_factors "0.9 1.0 1.1" --train_set train --valid_set dev --test_sets test
echo "################STAGE3###################"
./asr.sh --stage 3 --stop_stage 3 --train_set train --valid_set dev --test_sets test --nj 30 --audio_format wav
echo "################STAGE4###################"
./asr.sh --stage 4 --stop_stage 4 --train_set train --valid_set dev --test_sets test
echo "################STAGE5###################"
./asr.sh --stage 5 --stop_stage 5 --train_set train --valid_set dev --test_sets test --nbpe 256
echo "################STAGE6###################"
./asr.sh --stage 6 --stop_stage 6 --train_set train --valid_set dev --test_sets test --nj 30 --lm_config "conf/train_lm_transformer2.yaml" --nbpe 256
echo "DONE!!!!!!!"
