# Usage

## Dataset preparation

Please download the VCC2020 dataset following the instruction from the organizers. Then, correctly place the data. The default is `db_root=downloads/official_v1.0_training`

In VCC2020, there are 70 training utterances for each speaker. The list files that split train/dev sets are in `conf/lists`. Default is 60/10 split. Feel free to adjust them.

## Target speaker finetuning

Execute the main script to finetune `TMF1`-dependent TTS:

```
$ ./run_task2.sh --stop_stage 5 \
  --spk TMF1 --lang Man \
  --trans_type phn \
  --pretrained_model_name tts1_en_zh
```

Please make sure the parameters are carefully set:

1. Specify the `spk` and `lang`.
2. Specify `trans_type` depending on `lang`. If `lang` is `Man`, set to `phn`; if `Ger/Fin`, set to `char`.
3. Specify `pretrained_model_dir`. If you wish to use the pretrained model we provide, leave `pretrained_model_dir` to the default value (`downloads`). If it does not exist, it will be automatically downloaded. If you wish to use your own trained model, set `pretrained_model_dir` to (`../tts1_en_[de/fi/zh]`).
4. Specify `pretrained_model_name` depending on `lang`: `tts1_en_[de/fi/zh]`.
5. Specify `stop_stage` to no larger than 5.

With this main script, a full procedure of TTS training is performed:

- Stage -1: Pretrained model downloading, including the pretrained TTS and PWG models.
- Stage 0: Data preparation. The rules of transcription parsing depends on `lang` and is consistent with the parsing performed in TTS pretraining.
- Stage 1: Feature extraction. The features are normalized using the stats calculated in the TTS pretraining.
- Stage 2: JSON format data preparation. The tokens are indexed using the dictionary built in the TTS pretraining.
- Stage 3: X-vector extraction. This is based on the pre-trained, Kaldi-based x-vector extraction.
- Stage 4: Model training.
- Stage 5: Decoding. We decode the development set, which is the same language.


## Conversion phase

Execute the main script to convert `SEF1` to `TMF1`:

```
$ ./run_task2.sh --stage 11 \
  --srcspk SEF1 --trgspk TMF1 --trans_type char \
  --tts_model_dir exp/<expdir> \
  --pretrained_model tts1_en_zh \
  --voc PWG
```

Please make sure the parameters are carefully set:

1. Specify the `srcspk` and `trgspk`.
2. Specify `trans_type` depending on the language of the TTS model.
3. Specify `tts_model_dir` to the exp
4. Specify `pretrained_model`. The dictionary for tokenization and stats for normalization are used.
5. Specify `voc`. If you wish to use the Griffin-Lim algorithm, set to `GL`; if wish to use the trained PWG, set to `PWG`. 
6. Specify `stage` to larger than 11.

With this main script, a full procedure of TTS training is performed:

- Stage 11: Source speech recognition using the ASR model.
- Stage 12: Decoding. This includes cleaning the recognition results, tokenization, and decoding mel filterbanks using the TTS model. Note that the average of all the x-vectors of each training utterance of the target speaker is used.
- Stage 13: Synthesis. The Griffin-Lim phase recovery algorithm or the trained PWG model can be used to convert the generated mel filterbanks back to waveform.
