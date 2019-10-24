# JVS (Japanese versatile speech) corpus

https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus

## tts1

JVS provides transcriptions in 漢字仮名交じり文. However, due to the  large number of vocabularies in Chinese characters, we convert the input transcription to more compact representation: kana or phoneme. For this reason, you will have to install the text processing frontend ([OpenJTalk](http://open-jtalk.sp.nitech.ac.jp/)) to run the recipe. Please try the following to install the dependencies:

```
cd ${MAIN_ROOT}/tools && make pyopenjtalk.done
```

or manually install dependencies by following the instruction in https://github.com/r9y9/pyopenjtalk if you are working on your own python environment.

Unlike the other `tts1` recipes, `jvs/tts1` will finetune the pretrained models, which was trained on `jsut/tts1`, since JVS Databases is too small to train the E2E-TTS model from scrach. The statistics of feature vectors `cmvn.ark` and the dictionary file `dict` are not created but pretrained model's ones will be used. Also, it is necessary to use exactly the same architecture to finetune the pretrained model so please be careful `train_confg` is the same as the pretrained model's one.
