# Usage

## Data split

In VCC2020, there are 70 training utterances for each speaker. The list files that split train/dev sets are in `local/lists`. To get the results from the original baseline, the default is 60/10 split. Feel free to adjust them.

## Flow

We provide pretrained TTS models for users to finetune the models manually, and we also provide finetuned models for users to directly use.

### 1-1. Option 1: Target speaker finetuning

#### Execution

Execute the main script to finetune `TMF1`-dependent TTS:

```
$ ./run.sh --stop_stage 6 \
  --spk TMF1 --lang Man \
  --trans_type phn \
  --pretrained_model_name tts1_en_zh
```

Please make sure the parameters are carefully set:

1. Specify the `spk` and `lang`.
2. Specify `trans_type` depending on `lang`. If `lang` is `Man`, set to `phn`; if `Ger/Fin`, set to `char`.
3. Specify `pretrained_model_dir`. If you wish to use the pretrained model we provide, leave `pretrained_model_dir` to the default value (`downloads`). If it does not exist, it will be automatically downloaded. If you wish to use your own trained model, set `pretrained_model_dir` to (`../tts1_en_[de/fi/zh]`).
4. Specify `pretrained_model_name` depending on `lang`: `tts1_en_[de/fi/zh]`.
5. Specify `voc`. If you wish to use the Griffin-Lim algorithm, set to `GL`; if wish to use the trained PWG, set to `PWG`.
6. Specify `stop_stage` to no larger than 6.

With this main script, a full procedure of TTS finetuning is performed:

- Stage -1: Pretrained model downloading, including the pretrained TTS and PWG models.
- Stage 0: Data preparation. The rules of transcription parsing depends on `lang` and is consistent with the parsing performed in TTS pretraining.
- Stage 1: Feature extraction. The features are normalized using the stats calculated in the TTS pretraining.
- Stage 2: JSON format data preparation. The tokens are indexed using the dictionary built in the TTS pretraining.
- Stage 3: X-vector extraction. This is based on the pre-trained, Kaldi-based x-vector extraction.
- Stage 4: Model training.
- Stage 5: Decoding. We decode the development set, which is the same language.
- Stage 6: Synthesis. The Griffin-Lim phase recovery algorithm or the trained PWG model can be used to convert the generated mel filterbanks back to the waveform.

### 1-2. Option 2: Download finetuned TTS model

Execute the following script to download finetuned model for `TMF1`:

```
$ ./run.sh --stage 10 --stop_stage 10 \
  --pretrained_model_name tts1_en_zh \
  --finetuned_model_name tts1_en_zh_TMF1
```

Please make sure the parameters are carefully set:

1. Must set `finetuned_model_name` to `tts1_en_[de/fi/zh]_[trgspk]`.
2. It is recommended to also set `pretrained_model_name` to avoid generating redundant config files.

### 2. Conversion

Execute the main script to convert `SEF1` to `TMF1`:

```
$ ./run.sh --stage 11 \
  --srcspk SEF1 --trgspk TMF1 --trans_type char \
  --tts_model_dir <expdir> \
  --pretrained_model_name tts1_en_zh
```

Please make sure the parameters are carefully set:

1. Specify the `srcspk` and `trgspk`.
2. Specify `trans_type` depending on the language of the TTS model.
3. Specify `tts_model_dir` to the finetuned TTS experiment directory. If you wish to use your own trained model, set to `exp/<expdir>`. If you wish to use the finetuned model we provide, set to, for example, `downloads/tts1_en_[de/fi/zh]_[trgspk]/exp/<expdir>`.
4. Specify `pretrained_model`. The dictionary for tokenization and stats for normalization are used. **Note that this is still necessary even if you choose to use the finetuned model we provide.**
5. Specify `voc`. If you wish to use the Griffin-Lim algorithm, set to `GL`; if wish to use the trained PWG, set to `PWG`.
6. Specify `stage` to larger than 11.

With this main script, conversion is performed as follows:

- Stage 11: Source speech recognition using the ASR model.
- Stage 12: Decoding. This includes cleaning the recognition results, tokenization, and decoding mel filterbanks using the TTS model. Note that the average of all the x-vectors of each training utterance of the target speaker is used.
- Stage 13: Synthesis. The Griffin-Lim phase recovery algorithm or the trained PWG model can be used to convert the generated mel filterbanks back to the waveform.
- Stage 14: Objective evaluation. MCD as well as CER and WER from an ASR engine wil be calculated.

## Notes

### Text normalization mismatch between ASR and TTS

The text cleaner functions converts all text into uppercase and perserves punctuations. Ex:

`<fi_FI> MAROKON MAATALOUSALA SAA OSAKSEEN ETUOIKEUTETTUA KOHTELUA, KUN SE VIE TUOTTEITAAN EUROOPPAAN`

However, in the output of the pretrained ASR, there is no punctuation:

`IN REALITY THE EUROPEAN PARLIAMENT IS PRACTISING DIALECTICS (SEF1_E30001)`

This will cause a small mismatch, but we think it should be acceptable.

### MCD

The MCD is calculated w.r.t. the speech of the bilingual target speakers. However, they can be accented, where the ultimate goal is to preserve the natural accent of the source speech. So, the MCD values should be regarded as a reference only and not a reliable performance measure of the model.
