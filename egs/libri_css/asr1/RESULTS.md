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

# Conformer results

Similar to the Transformer configuration, the only difference is E2E ASR model.
 - E2E ASR model is trained on LibriSpeech 960h data with SpecAugment
data augmentation and 3-way speed perturbation using
[this](https://github.com/espnet/espnet/tree/b3a6efef16df1b9ccb19477adeb08f0eed44ca0e/egs/librispeech/asr1) recipe.
It has Encoder-Decoder architecture with 12 layers in Conformer Encoder
and 6 layers in Transformer Decoder.
The following command can be used to run this experiment:
```shell
./run.sh \
	--asr_url "https://drive.google.com/uc?id=153jimILkdUcD5L6g2a3ekeG0AxYaU7Nh" \
	--asr_dir download/asr_librispeech_sp_conformer \
	--recog_model model.val10.avg.best
```

### Oracle Segmentation
|Subset|0L|0S|OV10|OV20|OV30|OV40|WER (avg.)|
|---|---|---|---|---|---|---|---|
|Dev|3.88|6.56|13.12|19.69|30.04|35.18|19.72|
|Eval|6.49|7.17|12.36|20.35|29.10|38.68|20.70|

### X-vector + Spectral Clustering Diarization
|Subset|0L|0S|OV10|OV20|OV30|OV40|WER (avg.)|
|---|---|---|---|---|---|---|---|
|Dev|16.55|11.90|24.74|28.85|35.73|44.05|28.38|
|Eval|14.17|15.02|20.26|29.49|35.91|41.97|27.58|

### X-vector + BHMM Diarization
|Subset|0L|0S|OV10|OV20|OV30|OV40|WER (avg.)|
|---|---|---|---|---|---|---|---|
|Dev|9.46|12.68|23.07|29.84|36.50|45.61|28.07|
|Eval|18.30|15.19|21.36|31.37|25.67|43.80|29.00|

### X-vector + Agglomerative Hierarchical Clustering Diarization
|Subset|0L|0S|OV10|OV20|OV30|OV40|WER (avg.)|
|---|---|---|---|---|---|---|---|
|Dev|11.06|15.94|29.93|32.32|39.14|47.72|31.28|
|Eval|21.56|17.13|23.92|34.94|38.95|45.81|31.74|
