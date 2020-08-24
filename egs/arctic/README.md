# CMU ARCTIC Databases

http://www.festvox.org/cmu_arctic/

## tts1

Unlike the other `tts1` recipes, `arctic/tts1` will finetune the pretrained models, which was trained on `mailabs/tts1`, since CMU ARCTIC Databases is too small to train the E2E-TTS model from scrach.

The statistics of feature vectors `cmvn.ark` and the dictionary file `dict` are not created but pretrained model's ones will be used.

Also, it is necessary to use exactly the same architecture to finetune the pretrained model so please be careful `train_confg` is the same as the pretrained model's one.

## vc1

We support acoustic-to-acoustic feature conversion based on parallel data with two kinds of models, which are based on Tacotron2 and Transformer, respectively.

A pretraining technique utilizing TTS can be employed to boost the performance, as described in [1]. We provide the base pretrained model, which was trained on the M-AILABS dataset (the en_US judy speaker was used, approx. 32 hrs of data).

Note that you still need to train from scratch (or fine-tune using the base pretrained model) the VC model by yourself.

## Reference

- [1] Huang, Wen-Chin, et al. "Voice transformer network: Sequence-to-sequence voice conversion using transformer with text-to-speech pretraining." arXiv preprint arXiv:1912.06813 (2019).
