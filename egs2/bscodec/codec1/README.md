# BSCodec
BSCodec is a codec that performs well across multiple domains, offering relatively lower bit rates and better reconstruction performance compared to previous general-purpose codecs.

It is based on DAC structure and band-split strategy, and was trained using the ESPnet codec training pipeline.

The model checkpoint is available at https://huggingface.co/anonymous-release/BSCodec/tree/main

## Results
Please refer to our paper for comprehensive evaluations. Below are selected results on reconstruction:

| Model - Codec | Model - VQ Method | Model - Bitrate | Speech - MCD↓ | Speech - PESQ↑ | Speech - STOI↑ | Speech - SPK_SIM↑ | Speech - UTMOS↑ | Sound - VISQOL↑ | Sound - Mel Dist.↓ | Sound - STFT Dist.↓ | Music - VISQOL↑ | Music - Mel Dist.↓ | Music - STFT Dist.↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EnCodec† | RVQ | 6.00 kbps | 5.94 | 2.715 | 0.939 | 0.865 | 3.038 | 4.240 | 0.485 | 0.940 | 4.410 | 0.435 | 0.980 |
| DAC | RVQ | 6.00 kbps | 5.40 | 2.915 | 0.934 | 0.751 | 3.356 | 4.085 | 0.452 | 0.874 | 4.201 | 0.439 | 0.974 |
| DAC | RVQ | 4.50 kbps | 5.50 | **2.726** | **0.925** | 0.734 | 3.201 | 4.055 | 0.463 | 0.880 | 4.171 | **0.449** | 0.979 |
| DAC | RVQ | 3.00 kbps | 5.74 | 2.397 | 0.905 | 0.686 | 2.869 | 3.990 | 0.485 | 0.893 | 4.105 | 0.472 | 0.993 |
| EnCodec | RVQ | 3.00 kbps | 6.49 | 2.048 | 0.901 | 0.771 | 2.305 | 4.085 | 0.531 | 0.978 | 4.262 | 0.481 | 1.014 |
| BSCodec | 5# VQ | 3.75 kbps | 5.08 | 1.961 | 0.894 | 0.810 | 2.515 | **4.245** | 0.463 | 0.800 | **4.326** | 0.464 | 0.892 |
| BSCodec | 3# SimVQ | 3.83 kbps | **5.05** | 2.544 | 0.920 | **0.852** | **3.360** | 4.234 | **0.456** | **0.794** | 4.298 | 0.461 | **0.888** |
| BSCodec | 2# SimVQ | 2.55 kbps | 5.42 | 2.429 | 0.916 | 0.783 | 3.304 | 4.137 | 0.470 | 0.846 | 4.166 | 0.479 | 0.916 |

## Guidelines

The model implementation is in `espnet2/gan_codec/bscodec`, and the recipe is in `egs2/bscodec/codec1`. If you want to run inference:

### Download the model

Download the `exp/` directory from [release](https://huggingface.co/anonymous-release/BSCodec/tree/main) and place it under `egs2/bscodec/codec1`


### Organize your test set

Download the `dump/` directory from [release](https://huggingface.co/anonymous-release/BSCodec/tree/main) and also place it under `egs2/bscodec/codec1`

You can check the `wav.scp` and `utt2num_samples`. Add the "wavid wavpath" pairs to `wav.scp` and "wavid wav_samples_in_24kHz" to `utt2num_samples`. If your wav files are not in 24kHz, please resample them to 24kHz.

### Run script

Check `egs2/bscodec/codec1/run.sh` and modify the `model` argument to try other models. Options include:

* `BSCodec_band_vq_5band`
* `BSCodec_band_simvq_3band`
* `BSCodec_band_simvq_2band`

The `test_sets` argument should match the name of your test set.

### Citations

```BibTex
@article{wang2025bscodec,
  title={BSCodec: A Band-Split Neural Codec for High-Quality Universal Audio Reconstruction},
  author={Wang, Haoran and Shi, Jiatong and Tian, Jinchuan and Li, Bohan and Yu, Kai and Watanabe, Shinji},
  journal={arXiv preprint arXiv:2511.06150},
  year={2025}
}
```