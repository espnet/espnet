# JVS (Japanese versatile speech) corpus

https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus

## tts1

Unlike the other `tts1` recipes, `jvs/tts1` will finetune the pretrained models, which was trained on `jsut/tts1`, since JVS Databases is too small to train the E2E-TTS model from scrach.

The statistics of feature vectors `cmvn.ark` and the dictionary file `dict` are not created but pretrained model's ones will be used.

Also, it is necessary to use exactly the same architecture to finetune the pretrained model so please be careful `train_confg` is the same as the pretrained model's one.
