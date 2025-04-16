MUSE Recipe for ESPnet-Codec

This README provides instructions and details for using the AMUSE dataset with ESPnet-Codec, designed to train and evaluate neural codecs for audio, music, and speech applications.

## Dataset Overview

The Audio, Music, and Speech Ensemble (AMUSE) dataset is a large-scale corpus comprising diverse high-quality data, originally introduced alongside the DAC neural codec. AMUSE consolidates multiple open-source datasets, offering comprehensive audio, music, and speech samples for versatile codec training.

### Dataset Sources

AMUSE includes data from:

- DAC's original training set (audio/music/speech)
- Additional speech corpora:
  - AISHELL3
  - Google i18n-TTS corpora
  - Mexican endangered languages datasets
- Additional music datasets:
  - OpenSinger
  - StyleSing111
  - M4Singer
  - Kiritan-singing
  - Oniku Kurumi Utagoe database
  - Natsume Singing database
  - Opencpop
  - ACE-KiSing (excluding original voices)
  - PJS
  - JSUT singing

Please check the [original ESPnet-Codec paper](http://arxiv.org/abs/2409.15897v2) for detailed references.

### Evaluation Sets

For evaluation purposes, AMUSE test sets include:

- Speech: 3,000 randomly subsampled utterances
- Audio: 3,000 randomly subsampled utterances
- Music: Complete music test sets, including PopCS, Ofuton P Utagoe, ACE-KiSing-original

## Supported Neural Codecs

The recipe currently supports five neural codec architectures:

- SoundStream
- Encodec
- DAC
- FunCodec
- HiFi-Codec

Pre-trained codec models are available in our ESPnet neural codecs collection on Hugging Face:

- [ESPnet Neural Codecs Collection](https://huggingface.co/collections/espnet/neural-codecs-67cb8c96859c53a6131a85ec)
- [ESPnet Codec Survey Pre-trained Models](https://huggingface.co/collections/espnet/codec-survey-pre-trained-models-67ce8e09568b741d1c4483c8)

## Evaluation Metrics

Codec performance is comprehensively assessed using VERSA (Versatile Speech and Audio Evaluation toolkit), covering metrics including but not limited to:

- **Intrusive Metrics**: PESQ, STOI, SI-SNR, VISQOL
- **Non-intrusive Metrics**: DNSMOS, UTMOS, PLCMOS
- **Other Perceptual Metrics**: CER/WER, Speaker similarity (SPK-SIM)

Please check the VERSA page for more information (https://github.com/wavlab-speech/versa)

## Usage Instructions

### Step-by-Step Recipe

1. **Prepare data**: Prepare the AMUSE dataset by puting all wav files from the above datasets in `wav.scp` format.
2. **Train models**: Follow the provided training configurations and scripts for each codec.
3. **Model inference**: Generate discrete codec tokens or audio outputs using trained models.
4. **Evaluation**: Use VERSA to evaluate and analyze the quality and effectiveness of codec outputs.

Detailed scripts for each stage can be found in this recipe.

## References

For detailed methodology and additional context, please refer to the [original ESPnet-Codec paper](http://arxiv.org/abs/2409.15897v2).
