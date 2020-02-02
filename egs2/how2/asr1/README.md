```
HOW TO GET AND USE HOW2:

1. Go to https://github.com/srvk/how2-dataset and follow instructions.
For ASR, the following parts should be selected when filling form:
   (audio) fbank+pitch features in Kaldi scp/ark format
   (en) English text

First part (feats), contains directory 'fbank_pitch_181516'.
Second part (text) contains 'how2-300h-v1' with following directories:
   how2-300h-v1
   |_ data/
      |_ val
      |_ train
      |_ dev5
   |_ features/

2. Set up 'HOW2_FEATS' and 'HOW_TEXT' in db.sh to where you put 'how-300h-v1'
   and 'fbank_pitch_181516' directories.
```