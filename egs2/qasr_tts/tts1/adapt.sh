# Copyright 2021 Massa Baali

pretrained_model=$1
# From data preparation to statistics calculation
./run.sh --stage 1 --stop_stage 5

# download pretrained model kan-bayashi/ljspeech_tts_train_transformer_raw_char_tacotron_train.loss.ave
wget ${pretrained}

# Replace token list with pretrained model's one
pyscripts/utils/make_token_list_from_config.py pretrained_model_path/exp/ljspeech_tts_train_transformer_raw_char_tacotron/config.yaml
# tokens.txt is created in model directory
mv dump/token_list/ljspeech_tts_train_transformer_raw_char_tacotron/tokens.{txt,txt.bak}
ln -s pretrained_dir/exp/ljspeech_tts_train_transformer_raw_char_tacotron/tokens.txt dump/token_list/new_model

# Train the model 
./run.sh --stage 6 --train_config conf/tuning/finetune_transformer.yaml --train_args \ 
"--init_param pretrained_model/exp/tts_train_transformer_raw_char_tacotron/train.loss.ave_5best.pth":::tts.enc.embed \
--tag finetune_pretrained_transformers

# Now the trained model above will be used as a teacher model for the Non-AR model FastSpeech2 
# Prepare durations file
./run.sh --stage 7  --tts_exp exp/tts_finetune_pretrained_transformers \
    --inference_args "--use_teacher_forcing true" \
    --test_sets "tr_no_dev dev eval1"
    
# Since fastspeech2 requires extra feature calculation, run from stage 5.
./run.sh --stage 5 \
    --train_config conf/tuning/train_conformer_fastspeech2.yaml \
    --teacher_dumpdir exp/tts_finetune_pretrained_transformers/decode_use_teacher_forcingtrue_train.loss.ave \
    --tts_stats_dir exp/tts_finetune_pretrained_transformers/decode_use_teacher_forcingtrue_train.loss.ave/stats \
    --write_collected_feats true
