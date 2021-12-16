# RESULTS
## Environments
- date: `Mon Jan 25 19:16:45 CST 2021`
- python version: `3.6.3 |Anaconda, Inc.| (default, Nov 20 2017, 20:41:42)  [GCC 7.2.0]`
- espnet version: `espnet 0.9.7`
- pytorch version: `pytorch 1.6.0`
- Git hash: `dcaba2585e28b85c815807165ba9953565ee8694`
  - Commit date: `Thu Jan 21 21:26:59 2021 +0800`

## enh_train_raw
- Model link: https://zenodo.org/record/4480771/files/enh_train_raw_valid.si_snr.ave.zip?download=1
- config: ./conf/train.yaml
- sample_rate: 8k
- min_or_max: min

|dataset|STOI|SAR|SDR|SIR|
|---|---|---|---|---|
|enhanced_dev|0.85|11.10|10.67|22.65|
|enhanced_test|0.85|10.92|10.42|22.08|
