# JSUT

Corpus information: https://sites.google.com/site/shinnosuketakamichi/publication/jsut

## asr1

TO BE FILLED.

## tts1

This is a text-to-speech recipe for JSUT. As similar to the [ljspeech recipe](../ljspeech/), Tacotron 2 and Transformer are available for the end-to-end models.

JSUT provides transcriptions in 漢字仮名交じり文. However, due to the  large number of vocabularies in Chinese characters, we convert the input transcription to more compact representation: kana or phoneme. For this reason, you will have to install the text processing frontend ([OpenJTalk](http://open-jtalk.sp.nitech.ac.jp/)) to run the recipe. Please try the following to install the dependencies:

```
cd ${MAIN_ROOT}/tools && make pyopenjtalk.done
```

or manually install dependencies by following the instruction in https://github.com/r9y9/pyopenjtalk if you are working on your own python environment.
