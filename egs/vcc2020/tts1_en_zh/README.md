# Usage

## Dataset preparation

In this recipe we use the following two datasets:

- The M-AILABS dataset contains speech from English speakers. Please download the M-AILABS dataset in the original recipe.
- The CSMSC dataset contains speech of a Mandarin speaker. Please download the CSMSC dataset in the original recipe.

## Requirement

Please install the `pypinyin` package for Mandarin character to pinyin conversion.

## Execution

Execute the main script to pretrain the TTS model:

```
$ ./run.sh
```

With this main script, a full procedure of TTS pretraining is performed:

- Stage 0: Data preparation. This includes transcription parsing.
- Stage 1: Feature extraction. The features are normalized using the stats calculated using the pooled dataset.
- Stage 2: JSON format data preparation.
- Stage 3: X-vector extraction. This is based on the pre-trained, Kaldi-based x-vector extraction.
- Stage 4: Model training.
- Stage 5: Decoding and synthesis. We decode the development set and synthesize the waveform using the Griffin-Lim algorithm for quick performance check.

## Notes

- Since Mandarin characters are ideographic (not directly associated with pronunciations like other phonetic languages), we convert them into pinyin, the romanization system for Mandarin. Because of this, we also convert English into phoneme.
- An extra language token is added at the beginning of each utterance. E.g. `<en_US> BUT JUST NOW MY LOVING TYRANTS WON'T ALLOW ME.`
