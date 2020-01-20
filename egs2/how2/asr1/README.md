```
HOW TO GET AND USE HOW2:

1. Go to https://github.com/srvk/how2-dataset and follow instructions.
For ASR, the following parts should be selected in form:
   (audio) fbank+pitch features in Kaldi scp/ark format
   (en) English text

Second part (text) contains 'how2-300h-v1' with following directories:
   how2-300h-v1
   |_ data/
      |_ val
      |_ train
      |_ dev5
   |_ features/
First part (feats), contains directory 'fbank_pitch_181516'.

2. Copy 'fbank_pitch_181516' contents into 'how2-300h-v1/features.

3. Set up 'HOW2' in db.sh to where you put 'how-300h-v1' directory.

----

RECIPE NOTES:

Transcriptions are first generated from subtitles, text and audio are then
re-aligned using a Kaldi's GMM/HMM model trained on WSJ. Thus, the transcriptions
contains numbers, punctuations, symbols, ..., which is not best-suited
for character-based models.

Current recipe contains temporay normalization scripts and text replacements files.
Scripts were written based on replacement rules generated automatically with tools
not provided here.
It should be replaced soon by a more general perl script, similar to espnet1.
```