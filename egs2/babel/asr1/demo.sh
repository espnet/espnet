#!/bin/bash
. ./path.sh

python3 -m espnet2.bin.asr_inference \
	--ngpu "0" \
	--data_path_and_name_and_type "demo/demo.scp,speech,sound" \
	--key_file "demo/demo.scp" \
	--asr_train_config "exp/asr_train_asr_raw_char/config.yaml" \
	--asr_model_file "exp/asr_train_asr_raw_char/valid.acc.ave.pth" \
	--output_dir "demo/transcribed" \
	--config "conf/decoder_asr.yaml" \
	--lm_train_config "exp/lm_train_lm_char/config.yaml" \
	--lm_file "exp/lm_train_lm_char/20epoch.pth"
