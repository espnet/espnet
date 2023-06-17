# Talrómur 2 recipe

This is a recipe for the [Talrómur 2 corpus](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/167), which is an Icelandic speech corpus intended for multi-speaker TTS development, containing studio recordings of speech from 40 Icelandic speakers, 20 male and 20 female. 

## Corpus
The corpus contains 56,225 studio-recorded single-sentence audio clips.Each speaker in the corpus contributes 929 and 1879 clips.

| Male voices | Female voices |
|---|---|
| s124, s176, s178, s181, s188 | s146, s180, s186, s208, s209 |
| s206, s220, s225, s234, s235 | s214, s215, s221, s264, s268 |
| s157, s162, s216, s222, s223 | s169, s185, s187, s200, s226 |
| s231, s236, s240, s250, s273 | s228, s247, s251, s256, s258 |

Since this corpus is intended for multi-speaker and speaker-adaptive tts no attempt is made to create single-speaker TTS models out of the individual voices in the corpus.

A more detailed description of the corpus may be found in its [README file (download link)](https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/167/README.md) and in [the official repository](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/167)

## Dependencies
This recipe relies on the [Ice-G2P](https://github.com/grammatek/ice-g2p) package for g2p transcriptions. This can be done by running the installation script `installers/install_ice_g2p.sh` from the tools directory. e.g. from this directory you would do
```
cd ../../../tools
installers/install_ice_g2p.sh
```

## Usage
The recipe provides 2 training scripts: train_multi_speaker_fastspeech2.sh and train_multi_speaker_tacotron2.sh. In order to train a FastSpeech 2 model with the current implementation, you need to have a teacher model for phoneme durations. Therefore for a given speaker, you need to have already trained a Tacotron model in order to train a Fastspeech model.

First download the data by running the following:
```
. ./db.sh
local/data_download.sh ${TALROMUR2}
```
Now, to train a Tacotron 2 model, simply run `./train_multi_speaker_tacotron2.sh`
Once a Tacotron model has been trained, you can run `./train_fastspeech2.sh` to obtain a FastSpeech 2 model. 


---
## Pretrained models
Training outputs are present on Huggingface for each speaker:

A fully trained x-vector based Tacotron2 model has been uploaded to HuggingFace: [Link](https://huggingface.co/espnet/talromur2_xvector_tacotron2).

This Tacotron2 model was trained with slightly modified parameters for the ice-g2p Transcriber object, defined within `IsG2P` in `espnet2/text/phoneme_tokenizer.py`.

`word_sep=".", syllab_symbol=""` was used instead of the default `word_sep=",", syllab_symbol="."`. When using the model for inference, these nonstandard parameters must be used when producing the phonemized inputs.

---

## Acknowledgments
This project was funded by the Language Technology Programme for Icelandic 2019-2023. The programme, which is managed and coordinated by Almannarómur, is funded by the Icelandic Ministry of Education, Science and Culture.