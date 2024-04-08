# Usage

## Data split

In VCC2020, there are 70 training utterances for each speaker. The list files that split train/dev sets are in `local/lists`. To get the results from the original baseline, the default is 60/10 split. Feel free to adjust them.

## Flow

We provide pretrained TTS models for users to finetune the models manually, and we also provide finetuned models for users to directly use.

### 1-1. Option 1: Target speaker finetuning

#### Pretrained TTS model

Task 1 is monolingual (English) VC, so we can utilize the rich English resource thanks to the community. We strongly recommend users to use the pretrained TTS model trained on LibriTTS provided by ESPnet. However, if you wish to pretrain your own model, please refer to the [LibriTTS](https://github.com/espnet/espnet/tree/master/egs/libritts/tts1) recipe.

#### Execution

Execute the main script to finetune `TEF1`-dependent TTS:

```
$ ./run.sh --stop_stage 6 \
  --spk TEF1 \
  --pretrained_model_name tts1
```

Please make sure the parameters are carefully set:

1. Specify the `spk`.
2. Specify `pretrained_model_dir`. If you wish to use the pretrained model we provide, leave `pretrained_model_dir` to the default value (`downloads`). If it does not exist, it will be automatically downloaded. If you wish to use your own trained model, set `pretrained_model_dir` to, for example, `../libritts`.
3. Specify `pretrained_model_name`. Default: `tts1`.
4. Specify `voc`. If you wish to use the Griffin-Lim algorithm, set to `GL`; if wish to use the trained PWG, set to `PWG`.
5. Specify `stop_stage` to no larger than 6.

With this main script, a full procedure of TTS finetuning is performed:

- Stage -1: Pretrained model downloading, including the pretrained TTS and PWG models.
- Stage 0: Data preparation. The rules of transcription parsing is consistent with the parsing performed in TTS pretraining.
- Stage 1: Feature extraction. The features are normalized using the stats calculated in the TTS pretraining.
- Stage 2: JSON format data preparation. The tokens are indexed using the dictionary built in the TTS pretraining.
- Stage 3: X-vector extraction. This is based on the pre-trained, Kaldi-based x-vector extraction.
- Stage 4: Model training.
- Stage 5: Decoding. We decode the development set.
- Stage 6: Synthesis. The Griffin-Lim phase recovery algorithm or the trained PWG model can be used to convert the generated mel filterbanks back to the waveform.

### 1-2. Option 2: Download finetuned TTS model

Execute the following script to download finetuned model for `TEF1`:

```
$ ./run.sh --stage 10 --stop_stage 10 \
  --pretrained_model_name tts1 \
  --finetuned_model_name tts1_TEF1
```

Please make sure the parameters are carefully set:

1. Must set `finetuned_model_name` to `tts1_[trgspk]`.
2. It is recommended to also set `pretrained_model_name` to avoid generating redundant config files.

### 2. Conversion

Execute the main script to convert `SEF1` to `TEF1`:

```
$ ./run.sh --stage 11 \
  --srcspk SEF1 --trgspk TEF1 \
  --tts_model_dir <expdir> \
  --pretrained_model_name tts1
```

Please make sure the parameters are carefully set:

1. Specify the `srcspk` and `trgspk`.
2. Specify `tts_model_dir` to the finetuned TTS experiment directory. If you wish to use your own trained model, set to `exp/<expdir>`. If you wish to use the finetuned model we provide, set to, for example, `downloads/tts1_[trgspk]/exp/<expdir>`.
3. Specify `pretrained_model`. The dictionary for tokenization and stats for normalization are used. **Note that this is still necessary even if you choose to use the finetuned model we provide.**
4. Specify `voc`. If you wish to use the Griffin-Lim algorithm, set to `GL`; if wish to use the trained PWG, set to `PWG`.
5. Specify `stage` to larger than 11.

With this main script, conversion is performed as follows:

- Stage 11: Source speech recognition using the ASR model.
- Stage 12: Decoding. This includes cleaning the recognition results, tokenization, and decoding mel filterbanks using the TTS model. Note that the average of all the x-vectors of each training utterance of the target speaker is used.
- Stage 13: Synthesis. The Griffin-Lim phase recovery algorithm or the trained PWG model can be used to convert the generated mel filterbanks back to the waveform.
- Stage 14: Objective evaluation. MCD as well as CER and WER from an ASR engine wil be calculated.

## Notes

### Text normalization mismatch between ASR and TTS

LibriTTS itself provides normalized text, and the pretrained model is directly trained on such text. Ex:

`100_121669_000001_000000 Tom, the Piper's Son`

However, the output of the pretrained ASR is always uppercased, and there is no punctuation:

`IN REALITY THE EUROPEAN PARLIAMENT IS PRACTISING DIALECTICS (SEF1_E30001)`

Considering that recovering the true case is not easy, and training ASR models with the true case is not so successful, we simply lowercase the ASR results. This will cause a small mismatch, but we think it should be acceptable.
