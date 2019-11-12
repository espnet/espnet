# v.0.5.3 / FastSpeech

- Silence trimming
- FTT in points: 1024
- Shift in points: 256
- Frequency limit: 80-7600
- Fast-GL 64 iters

## Environments

- date: `Sat Oct 12 21:36:17 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `36321a262b8dbb6c13297d20e050192624366517`
  - Commit date: `Sat Oct 12 13:24:10 2019 +0900`

## Models (it_IT, female)

- model link: https://drive.google.com/open?id=1EJpluSMxRh4muJnGdwehXdUUWjIxWvNZ
- training config file: `conf/tuning/train_fastspeech.v2.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/it_IT_lisa_train_trim/cmvn.ark`
- e2e file: `exp/it_IT_lisa_train_trim_pytorch_train_fastspeech.v2/results/model.last1.avg.best`
- e2e JSON file: `exp/it_IT_lisa_train_trim_pytorch_train_fastspeech.v2/results/model.json`
- dict file: `data/lang_1char/it_IT_lisa_train_trim_units.txt`

## Sample (it_IT, female)

https://drive.google.com/open?id=13I5V2w7deYFX4DlVk1-0JfaXmUR2rNOv


# v.0.5.3 / Transformer

- Silence trimming
- FTT in points: 1024
- Shift in points: 256
- Frequency limit: 80-7600
- Fast-GL 64 iters

## Environments

- date: `Sun Sep 29 21:20:05 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.1`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `6b2ff45d1e2c624691f197014b8fe71a5e70bae9`
  - Commit date: `Sat Sep 28 14:33:32 2019 +0900`

## Models (en_US, female)

- model link: https://drive.google.com/open?id=1rHQMMjkSoiX3JX2e70MKUKSrxHGwhmRb
- training config file: `conf/tuning/train_pytorch_transformer.v1.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/en_US_judy_train_trim/cmvn.ark`
- e2e file: `exp/en_US_judy_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.last1.avg.best`
- e2e JSON file: `exp/en_US_judy_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.json`
- dict file: `data/lang_1char/en_US_judy_train_trim_units.txt`

## Samples (en_US, female)

https://drive.google.com/open?id=1dj_e7evrqY4EYes1XWiPx52qwThe-zlY

## Models (en_UK, female)

- model link: https://drive.google.com/open?id=1iXdQv_YGD9VG1dR_xCjSkX6A4HkrpTbF
- training config file: `conf/tuning/train_pytorch_transformer.v1.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/en_UK_elizabeth_train_trim/cmvn.ark`
- e2e file: `exp/en_UK_elizabeth_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.last1.avg.best`
- e2e JSON file: `exp/en_UK_elizabeth_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.json`
- dict file: `data/lang_1char/en_UK_elizabeth_train_trim_units.txt`

## Samples (en_UK, female)

https://drive.google.com/open?id=1sscjjP3ks5eO2VAlqWRpVmDUqP2kIoUH

## Models (es_ES, female)

- model link: https://drive.google.com/open?id=1LF_zmZML-XaTYiyN3wensvrqlaTxat3p
- training config file: `conf/tuning/train_pytorch_transformer.v1.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/es_ES_karen_train_trim/cmvn.ark`
- e2e file: `exp/es_ES_karen_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.last1.avg.best`
- e2e JSON file: `exp/es_ES_karen_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.json`
- dict file: `data/lang_1char/es_ES_karen_train_trim_units.txt`

## Samples (es_ES, female)

https://drive.google.com/open?id=1sZitQfhKlePyTPFEbZcpMMmGHGT8Ued3

## Models (de_DE, female)

- model link: https://drive.google.com/open?id=1Yoz7UY_Fj2m5sUGVnaVNDhAIveQ4JsBY
- training config file: `conf/tuning/train_pytorch_transformer.v1.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/de_DE_eva_train_trim/cmvn.ark`
- e2e file: `exp/de_DE_eva_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.last1.avg.best`
- e2e JSON file: `exp/de_DE_eva_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.json`
- dict file: `data/lang_1char/de_DE_eva_train_trim_units.txt`

## Samples (de_DE, female)

https://drive.google.com/open?id=1mfj3ztX8DuYyn6tO-9fM2YVaXrw14b7S

## Models (it_IT, female)

- model link: https://drive.google.com/open?id=18VuXuTEuvQWsuq5vz-eFgXclauFd-XXD
- training config file: `conf/tuning/train_pytorch_transformer.v1.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/it_IT_lisa_train_trim/cmvn.ark`
- e2e file: `exp/it_IT_lisa_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.last1.avg.best`
- e2e JSON file: `exp/it_IT_lisa_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.json`
- dict file: `data/lang_1char/it_IT_lisa_train_trim_units.txt`

## Samples (it_IT, female)

https://drive.google.com/open?id=16LTgCVlS4m2dn7VMHyC2zW1Th9e1i6ie

## Models (fr_FR, female)

- model link: https://drive.google.com/open?id=1IBEICJtRypZOL7k6XWwRbV4v6QbYuXZi
- training config file: `conf/tuning/train_pytorch_transformer.v1.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/fr_FR_ezwa_train_trim/cmvn.ark`
- e2e file: `exp/fr_FR_ezwa_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.last1.avg.best`
- e2e JSON file: `exp/fr_FR_ezwa_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.json`
- dict file: `data/lang_1char/fr_FR_ezwa_train_trim_units.txt`

## Samples (fr_FR, female)

https://drive.google.com/open?id=1NQ4v-x2LyYl6yAzo3EYGkLlWMeYPWsir

## Models (en_US, male)

- model link: https://drive.google.com/open?id=1zv9GwhhBW32a6RM5wHzjqRxkkv9IrXTL
- training config file: `conf/tuning/train_pytorch_transformer.v1.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/en_US_elliot_train_trim/cmvn.ark`
- e2e file: `exp/en_US_elliot_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.last1.avg.best`
- e2e JSON file: `exp/en_US_elliot_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.json`
- dict file: `data/lang_1char/en_US_elliot_train_trim_units.txt`

## Sample (en_US, male)

https://drive.google.com/open?id=1AxGtcwbX07iScZvZPyQxnxM89nwh5oI2

## Models (es_ES, male)

- model link: https://drive.google.com/open?id=1azezvD4qGTGoGoGVMkXINoHAkXnCEEPI
- training config file: `conf/tuning/train_pytorch_transformer.v1.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/es_ES_tux_train_trim/cmvn.ark`
- e2e file: `exp/es_ES_tux_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.last1.avg.best`
- e2e JSON file: `exp/es_ES_tux_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.json`
- dict file: `data/lang_1char/es_ES_tux_train_trim_units.txt`

## Sample (es_ES, male)

https://drive.google.com/open?id=11k2iEWUa0IXHEgRBPHxmNnptQ48QVrf3

## Models (fr_FR, male)

- model link: https://drive.google.com/open?id=1Pz80HaVyFzk_VSHl6Zoc1dBTUEXUJJoI
- training config file: `conf/tuning/train_pytorch_transformer.v1.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/fr_FR_bernard_train_trim/cmvn.ark`
- e2e file: `exp/fr_FR_bernard_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.last1.avg.best`
- e2e JSON file: `exp/fr_FR_bernard_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.json`
- dict file: `data/lang_1char/fr_FR_bernard_train_trim_units.txt`

## Samples (fr_FR, male)

https://drive.google.com/open?id=1mRBszbkb8gx1MD_Ffagu--CyOg93_mXO

## Models (it_IT, male)

- model link: https://drive.google.com/open?id=1j7pcClXiRt4T68DcOZ2-nf7IWAIeV1E-
- training config file: `conf/tuning/train_pytorch_transformer.v1.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/it_IT_riccardo_train_trim/cmvn.ark`
- e2e file: `exp/it_IT_riccardo_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.last1.avg.best`
- e2e JSON file: `exp/it_IT_riccardo_train_trim_pytorch_train_pytorch_transformer.v1.single/results/model.json`
- dict file: `data/lang_1char/it_IT_riccardo_train_trim_units.txt`

## Samples (it_IT, male)

https://drive.google.com/open?id=1QLtB4zo0lwrGZyEXJ9IL-rgmH7HV8u0i

## Models (de_DE, male)

- model link: https://drive.google.com/open?id=1EXjL5ogjEgUj_hMv1q5pOKg3Q9KHSHza
- training config file: `conf/tuning/train_pytorch_transformer.v1.single_batch-bins607200.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/de_DE_karlsson_train_trim/cmvn.ark`
- e2e file: `exp/de_DE_karlsson_train_trim_pytorch_train_pytorch_transformer.v1.single_batch-bins607200/results/model.last1.avg.best`
- e2e JSON file: `exp/de_DE_karlsson_train_trim_pytorch_train_pytorch_transformer.v1.single_batch-bins607200/results/model.json`
- dict file: `data/lang_1char/de_DE_karlsson_train_trim_units.txt`

## Samples (de_DE, male)

https://drive.google.com/open?id=1J5_EZX54LUwuTVkwmNZ56E4oWdyB8MKF

# v.0.5.3 / Tacotron 2

- Silence trimming
- FTT in points: 1024
- Shift in points: 256
- Frequency limit: 80-7600
- Fast-GL 64 iters

## Environments

- date: `Sun Sep 29 21:20:05 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.1`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `6b2ff45d1e2c624691f197014b8fe71a5e70bae9`
  - Commit date: `Sat Sep 28 14:33:32 2019 +0900`

## Models (en_UK, female)

- model link: https://drive.google.com/open?id=1iOwvCx6wX5_qCmHZSX_vCd_ZYn-B5akh
- training config file: `conf/tuning/train_pytorch_tacotron2.v3.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/en_UK_elizabeth_train_trim/cmvn.ark`
- e2e file: `exp/en_UK_elizabeth_train_trim_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best`
- e2e JSON file: `exp/en_UK_elizabeth_train_trim_pytorch_train_pytorch_tacotron2.v3/results/model.json`
- dict file: `data/lang_1char/en_UK_elizabeth_train_trim_units.txt`

## Samples (en_UK, female)

https://drive.google.com/open?id=1_2w48ZLqPrJQv6VUQaTLkrQwD9pYx4R7

## Models (it_IT, female)

- model link: https://drive.google.com/open?id=1iF8avkisNd1JChmPf7mrR00kulT_mx_r
- training config file: `conf/tuning/train_pytorch_tacotron2.v3.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/it_IT_lisa_train_trim/cmvn.ark`
- e2e file: `exp/it_IT_lisa_train_trim_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best`
- e2e JSON file: `exp/it_IT_lisa_train_trim_pytorch_train_pytorch_tacotron2.v3/results/model.json`
- dict file: `data/lang_1char/it_IT_lisa_train_trim_units.txt`

## Samples (it_IT, female)

https://drive.google.com/open?id=1AGD4jD1fPHgv3VDLqNgzd9fyEbBBul7I

## Models (fr_FR, female)

- model link: https://drive.google.com/open?id=1LbD_Y5XCwO3ealILCgal9xz3dF_G_tE1
- training config file: `conf/tuning/train_pytorch_tacotron2.v3.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/fr_FR_ezwa_train_trim/cmvn.ark`
- e2e file: `exp/fr_FR_ezwa_train_trim_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best`
- e2e JSON file: `exp/fr_FR_ezwa_train_trim_pytorch_train_pytorch_tacotron2.v3/results/model.json`
- dict file: `data/lang_1char/fr_FR_ezwa_train_trim_units.txt`

## Samples (fr_FR, female)

https://drive.google.com/open?id=1r5urqTs9JMr6tjTVfZwPRFX191hBHVRw

## Models (en_US, female)

- model link: https://drive.google.com/open?id=1cNrTa8Jxa3AYcap7jo0_RPBapiay3etG
- training config file: `conf/tuning/train_pytorch_tacotron2.v3.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/en_US_judy_train_trim/cmvn.ark`
- e2e file: `exp/en_US_judy_train_trim_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best`
- e2e JSON file: `exp/en_US_judy_train_trim_pytorch_train_pytorch_tacotron2.v3/results/model.json`
- dict file: `data/lang_1char/en_US_judy_train_trim_units.txt`

## Samples (en_US, female)

https://drive.google.com/open?id=1H7W9jhbk6xsMfPCXLoE9rrBhXeUBP5It

## Models (es_ES, female)

- model link: https://drive.google.com/open?id=1k4uiIHYd1iXcAUkjMc99GIbV9IqM87Bu
- training config file: `conf/tuning/train_pytorch_tacotron2.v3.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/es_ES_karen_train_trim/cmvn.ark`
- e2e file: `exp/es_ES_karen_train_trim_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best`
- e2e JSON file: `exp/es_ES_karen_train_trim_pytorch_train_pytorch_tacotron2.v3/results/model.json`
- dict file: `data/lang_1char/es_ES_karen_train_trim_units.txt`

## Samples (es_ES, female)

https://drive.google.com/open?id=1louEChDdUSluEW3-VO9mMGxL5aR-OAxG

## Models (de_DE, female)

- model link: https://drive.google.com/open?id=15Lxw_smEvgAFcIWEi5wfC2ort8dAPYSk
- training config file: `conf/tuning/train_pytorch_tacotron2.v3.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/de_DE_eva_train_trim/cmvn.ark`
- e2e file: `exp/de_DE_eva_train_trim_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best`
- e2e JSON file: `exp/de_DE_eva_train_trim_pytorch_train_pytorch_tacotron2.v3/results/model.json`
- dict file: `data/lang_1char/de_DE_eva_train_trim_units.txt`

## Samples (de_DE, female)

https://drive.google.com/open?id=1wLySpRvIg50Cp8qIq-aO-Bw56BZg18iR

## Models (it_IT, male)

- model link: https://drive.google.com/open?id=10BQTEVmRwFUi9zNiSj28VFrgx8NCtOtu
- training config file: `conf/tuning/train_pytorch_tacotron2.v3.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/it_IT_riccardo_train_trim/cmvn.ark`
- e2e file: `exp/it_IT_riccardo_train_trim_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best`
- e2e JSON file: `exp/it_IT_riccardo_train_trim_pytorch_train_pytorch_tacotron2.v3/results/model.json`
- dict file: `data/lang_1char/it_IT_riccardo_train_trim_units.txt`

## Samples (it_IT, male)

https://drive.google.com/open?id=1o_mJqeQEUaI4Nqva6za-4HELzPKAvWLX

## Models (fr_FR, male)

- model link: https://drive.google.com/open?id=1sMmFil8FeZto-KJsmKGWg-TV-Jj4C9xe
- training config file: `conf/tuning/train_pytorch_tacotron2.v3.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/fr_FR_bernard_train_trim/cmvn.ark`
- e2e file: `exp/fr_FR_bernard_train_trim_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best`
- e2e JSON file: `exp/fr_FR_bernard_train_trim_pytorch_train_pytorch_tacotron2.v3/results/model.json`
- dict file: `data/lang_1char/fr_FR_bernard_train_trim_units.txt`

## Samples (fr_FR, male)

https://drive.google.com/open?id=1p1T-85pUxiJBRQRV76XkApky_2TQmK_6

## Models (es_ES, male)

- model link: https://drive.google.com/open?id=1988NUx2z6n6otFxiGyvDysuhPSqXroKd
- training config file: `conf/tuning/train_pytorch_tacotron2.v3.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/es_ES_tux_train_trim/cmvn.ark`
- e2e file: `exp/es_ES_tux_train_trim_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best`
- e2e JSON file: `exp/es_ES_tux_train_trim_pytorch_train_pytorch_tacotron2.v3/results/model.json`
- dict file: `data/lang_1char/es_ES_tux_train_trim_units.txt`

## Samples (es_ES, male)

https://drive.google.com/open?id=1eKiNCTnhtrU7ZvDKlIZjWDRBp8fdub0J

## Models (de_DE, male)

- model link: https://drive.google.com/open?id=1E3kGfR2P-P4jf3rByLoA2UtFZoXBEdza
- training config file: `conf/tuning/train_pytorch_tacotron2.v3_batch-bins1834560.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/de_DE_karlsson_train_trim/cmvn.ark`
- e2e file: `exp/de_DE_karlsson_train_trim_pytorch_train_pytorch_tacotron2.v3_batch-bins1834560/results/model.loss.best`
- e2e JSON file: `exp/de_DE_karlsson_train_trim_pytorch_train_pytorch_tacotron2.v3_batch-bins1834560/results/model.json`
- dict file: `data/lang_1char/de_DE_karlsson_train_trim_units.txt`

## Samples (de_DE, male)

https://drive.google.com/open?id=1-4hK-5d1Mf6tv0BTrokE8CnkDdYJbdnC

# v.0.3.0 trimming + 1024/256 + reduction_factor=1 + forward_ta attention

## Samples

https://drive.google.com/open?id=1q_66kyxVZGU99g8Xb5a0Q8yZ1YVm2tN0
