# Transformer results

This example demonstrates usage of pre-trained LibriSpeech model
for the decoding of [LibriCSS](https://arxiv.org/abs/2001.11482) dataset.
LibriCSS dataset contains long recordings with partially overlapped
speakers that is produced by replay of LibriSpeech utterances
and capture with far-field microphones.
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
|Dev|4.30|6.22|11.88|19.13|29.21|34.91|19.20|
|Eval|5.19|5.23|11.43|19.31|28.46|38.32|19.72|

### X-vector + Spectral Clustering Diarization
|Subset|0L|0S|OV10|OV20|OV30|OV40|WER (avg.)|
|---|---|---|---|---|---|---|---|
|Dev|15.29|13.24|23.01|28.24|34.80|44.90|28.07|
|Eval|12.56|13.88|19.76|29.83|35.11|41.75|27.01|

### X-vector + BHMM Diarization
|Subset|0L|0S|OV10|OV20|OV30|OV40|WER (avg.)|
|---|---|---|---|---|---|---|---|
|Dev|8.95|12.51|22.20|29.29|37.07|46.02|27.93|
|Eval|16.78|14.23|21.45|31.43|36.49|43.52|28.81|

### X-vector + Agglomerative Hierarchical Clustering Diarization
|Subset|0L|0S|OV10|OV20|OV30|OV40|WER (avg.)|
|---|---|---|---|---|---|---|---|
|Dev|11.66|17.68|25.18|31.72|38.78|48.66|30.85|
|Eval|20.29|16.55|24.55|34.64|40.05|46.50|31.90|
