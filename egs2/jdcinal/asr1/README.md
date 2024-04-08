# RESULTS
## Dataset
- Japanese Dialogue Corpus of Information Navigation and Attentive Listening Annotated with Extended ISO-24617-2 Dialogue Act Tags
  - Paper: http://www.lrec-conf.org/proceedings/lrec2018/pdf/464.pdf

## Environments
- date: `Sat Oct  9 20:44:56 EDT 2021`
- python version: `3.7.11 (default, Jul 27 2021, 14:32:16)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.3a4`
- pytorch version: `pytorch 1.9.0+cu102`
- Git hash: `2c8f7d884480ff46ecccd308d689405232ec345d`
  - Commit date: `Mon Oct 4 16:11:37 2021 -0400`


## Using Conformer based encoder-decoder and predicting transcript along with dialogue act
- ASR config: [conf/train_asr.yaml](conf/train_asr.yaml)
- token_type: word
- keep_nbest_models: 5

|dataset|Dialogue Act Classification (%)|
|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/test|67.4|
|decode_asr_asr_model_valid.acc.ave_10best/valid|67.8|
