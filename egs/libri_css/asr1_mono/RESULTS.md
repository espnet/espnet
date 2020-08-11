# Transformer results

This example demonstrates usage of pre-trained LibriSpeech model
for the decoding of [LibriCSS](https://arxiv.org/abs/2001.11482) dataset.
 - E2E ASR model is trained on LibriSpeech 960h data with SpecAugment
data augmentation using
[this](https://github.com/espnet/espnet/tree/47f51a77906c4c44d0da23da04e68676e4b931ab/egs/librispeech/asr1) recipe.
It has Transformer Encoder-Decoder architecture with 12 layers in Encoder
and 6 layers in Decoder.
 - External Transformer LM is trained on LibriSpeech LM training data using
[the same](https://github.com/espnet/espnet/tree/47f51a77906c4c44d0da23da04e68676e4b931ab/egs/librispeech/asr1) recipe.
It has Transformer architecture with 16 layers.

### Oracle Segmentation
|Subset|0L|0S|OV10|OV20|OV30|OV40|WER (avg.)|
|---|---|---|---|---|---|---|---|
|Dev|3.71|6.28|12.26|19.54|29.03|34.82|19.23|
|Eval|5.10|5.28|11.41|19.49|28.48|38.31|19.74|

### X-vector + Spectral Clustering Diarization
|Subset|0L|0S|OV10|OV20|OV30|OV40|WER (avg.)|
|---|---|---|---|---|---|---|---|
|Dev|14.19|12.23|26.20|29.29|35.21|44.23|28.44|
|Eval|12.40|14.14|20.24|29.50|35.26|41.90|27.11|
