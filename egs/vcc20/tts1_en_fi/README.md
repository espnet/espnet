# Usage

## Dataset preparation

In this recipe we use the following two datasets:

- The M-AILABS dataset contains English speech data.
- The CSS10 dataset contains Finnish speech data. This dataset has to be downloaded from Kaggle. Please download from [this link](https://www.kaggle.com/bryanpark/finnish-single-speaker-speech-dataset) and place it in the desired directory (default is `downloads`)

## Execution

Execute the main script to pretrain the TTS model:

```
$ ./run.sh
```

With this main script, a full procedure of TTS pretraining is performed:

- Stage -1: Data download. The M-AILABS dataset will be downloaded.
- Stage 0: Data preparation. This includes transcription parsing.
- Stage 1: Feature extraction. The features are normalized using the stats calculated using the pooled dataset.
- Stage 2: JSON format data preparation.
- Stage 3: X-vector extraction. This is based on the pre-trained, Kaldi-based x-vector extraction.
- Stage 4: Model training.
- Stage 5: Decoding and synthesis. We decode the development set and synthesize the waveform using the Griffin-Lim algorithm for a quick performance check.

## Notes

- All texts are capitalized, but the punctuations are preserved.
- An extra language token is added at the beginning of each utterance. E.g. `<en_US> BUT JUST NOW MY LOVING TYRANTS WON'T ALLOW ME.`
