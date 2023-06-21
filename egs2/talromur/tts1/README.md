# Talrómur recipe

This is a recipe for the [Talrómur corpus](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/104), which contains a large amount of studio recorded speech data from 8 Icelandic speakers, 4 male and 4 female. 


## Corpus
[This paper](https://aclanthology.org/2021.nodalida-main.50.pdf) describes the collection process and details of the corpus.
It is intended for single-speaker TTS development so this recipe requires a speaker_id parameter to be specified.
The IDs match the ones in the paper: (a,b,c,d,e,f,g,h), with speakers a, c, e, and g being female and speakers b, d, f, and h being male.

## Dependencies
This recipe relies on the [Ice-G2P](https://github.com/grammatek/ice-g2p) package for g2p transcriptions. This can be done by running the installation script `installers/install_ice_g2p.sh` from the tools directory. e.g. from this directory you would do
```
cd ../../../tools
installers/install_ice_g2p.sh
```

## Usage
The recipe provides 2 training scripts: train_fastspeech2.sh and train_tacotron2.sh. In order to train a FastSpeech 2 model with the current implementation, you need to have a teacher model for phoneme durations. Therefore for a given speaker, you need to have already trained a Tacotron model in order to train a Fastspeech model.

First download the data by running the following:
```
. ./db.sh
local/data_download.sh ${TALROMUR}
```
Now, to train a Tacotron 2 model, simply run `./train_tacotron2.sh <spk_id>` with the desired speaker ID.
Once a Tacotron model has been trained, you can run `./train_fastspeech2.sh <spk_id>` to obtain a FastSpeech 2 model. 


---
## Pretrained models
Training outputs are present on Huggingface for each speaker:

|Speaker ID| Tacotron model| Fastspeech model|
|---|---|---|
|a | [Link](https://huggingface.co/espnet/GunnarThor_talromur_a_tacotron2) |  [Link](https://huggingface.co/espnet/GunnarThor_talromur_a_fastspeech2)|
|b | [Link](https://huggingface.co/espnet/GunnarThor_talromur_b_tacotron2) |  [Link](https://huggingface.co/espnet/GunnarThor_talromur_b_fastspeech2)|
|c | [Link](https://huggingface.co/espnet/GunnarThor_talromur_c_tacotron2) |  [Link](https://huggingface.co/espnet/GunnarThor_talromur_c_fastspeech2)|
|d | [Link](https://huggingface.co/espnet/GunnarThor_talromur_d_tacotron2) |  [Link](https://huggingface.co/espnet/GunnarThor_talromur_d_fastspeech2)|
|e | [Link](https://huggingface.co/espnet/GunnarThor_talromur_e_tacotron2) |  [Link](https://huggingface.co/espnet/GunnarThor_talromur_e_fastspeech2)|
|f | [Link](https://huggingface.co/espnet/GunnarThor_talromur_f_tacotron2) |  [Link](https://huggingface.co/espnet/GunnarThor_talromur_f_fastspeech2)|
|g | [Link](https://huggingface.co/espnet/GunnarThor_talromur_g_tacotron2) |  [Link](https://huggingface.co/espnet/GunnarThor_talromur_g_fastspeech2)|
|h | [Link](https://huggingface.co/espnet/GunnarThor_talromur_h_tacotron2) |  [Link](https://huggingface.co/espnet/GunnarThor_talromur_h_fastspeech2)|

---

## Acknowledgments
This project was funded by the Language Technology Programme for Icelandic 2019-2023. The programme, which is managed and coordinated by Almannarómur, is funded by the Icelandic Ministry of Education, Science and Culture.
