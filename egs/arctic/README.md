# CMU ARCTIC Databases

http://www.festvox.org/cmu_arctic/

## tts1

Unlike the other `tts1` recipes, `arctic/tts1` will finetune the pretrained models, which was trained on `mailabs/tts1`, since CMU ARCTIC Databases is too small to train the E2E-TTS model from scrach.

The statistics of feature vectors `cmvn.ark` and the dictionary file `dict` are not created but pretrained model's ones will be used.

Also, it is necessary to use exactly the same architecture to finetune the pretrained model so please be careful `train_confg` is the same as the pretrained model's one.
