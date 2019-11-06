# ASJ Japanese Newspaper Article Sentences Read Speech Corpus (JNAS)

http://research.nii.ac.jp/src/JNAS.html

We assume that the database structure is as follows:

```
/database/JNAS
├── DOCS/
├── OriginalText/
├── readme_en.txt
├── readme_jp.txt
├── reference_en.pdf
├── reference_jp.pdf
├── Transcription/
├── WAVES_DT/
└── WAVES_HS/

/database/JNAS/DOCS/Test_set
├── ASJ_1998.pdf
├── IPA98_testset_100/
├── JNAS_testset_100/
├── JNAS_testset_500/
└── readme.txt
```

## asr1

TO BE FILLED.

## tts1

JNAS provides transcriptions in 漢字仮名交じり文. However, due to the  large number of vocabularies in Chinese characters, we convert the input transcription to more compact representation: kana or phoneme. For this reason, you will have to install the text processing frontend ([OpenJTalk](http://open-jtalk.sp.nitech.ac.jp/)) to run the recipe. Please try the following to install the dependencies:

```
cd ${MAIN_ROOT}/tools && make pyopenjtalk.done
```

or manually install dependencies by following the instruction in https://github.com/r9y9/pyopenjtalk if you are working on your own python environment.
