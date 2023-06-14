# Multilingual SUPERB (ML-SUPERB) benchmark

## Basic Information

### Language coverage (143 languages)
ML-SUPERB is designed to cover a wide range of languages, including both high-resource languages like English and endangered languages such as Totonac.

### Tasks
- Monolingual Track: monolingual speech recognition (ASR)
- Multilingual Track: multilingual ASR, language identification (LID), joint multilingual ASR+LID

### Dataset
Dataset are extracted from various multilingual sources. All sources are with either Creative Commons, MIT, GNU, or Free-BSD licenses, which are available for both industrial and academic research, permissively.

## Major usage guidelines

### Data download/setup

[Download link](https://drive.google.com/file/d/1zslKQwadZaYWXAmfBCvlos9BVQ9k6PHT/view?usp=sharing)

After download the dataset, please set the `MLSUPERB` to the data directory. The preparation will be automatically done in scripts for each tasks.

### Self-supervised model setup

ML-SUPERB utilizes [S3PRL](https://github.com/s3prl/s3prl) to support different self-supervised model, supporting both the current popular self-supervised models and customized models from users.

#### Existng self-supervised model

To use existing self-supervised model (e.g., models suppported by S3PRL), you can set the following configs for model training:

- For model directly supported by S3PRL, you may set the following argument in the yaml-style config files located in `conf/tuning`:
```
frontend: s3prl
frontend_conf:
    frontend_conf:
        upstream: hubert_large_ll60k
    download_dir: ./hub
    multilayer_feature: True
```
- For model from huggingface (still supported by S3PRL, such as HuBERT trained on specific languages), the config can be as:
```
frontend: s3prl
frontend_conf:
    frontend_conf:
        upstream: hf_hubert_custom
        path_or_url: <huggingface_id> # the huggingface ID (such as TencentGameMate/chinese-hubert-large)
    download_dir: ./hub
    multilayer_feature: True
```

**Note: If the upstream is changed, please change the input_size in the preencoder.**

#### Customized self-supervised model

To test the SSL model in ML-SUPERB that are not directly support in S3PRL, please first follow the guidelines at https://s3prl.github.io/s3prl/contribute/upstream.html to add new upstream models.

### Monolingual ASR

General steps to run tasks in monolingual ASR track are as follows:
- Step1: Following [downloading guide](https://github.com/espnet/espnet/blob/master/egs2/ml_superb/asr1/README.md#data-downloadsetup) to prepare the data
- Step2: Adding the training configurations for the desired model at `conf/tuning` (check examples `conf/tuning/train_asr_s3prl_single.yaml` and `conf/tuning/train_asr_fbank_single.yaml`) **Note: only the frontend/learning rate can be changed for the benchmark.**
- Step3: Training the model by calling
```
./run_mono.sh --asr_config <your_training_config>
```
- Step4: Results are at `exp/mono_<your_training_config>.log`

### Multilingual ASR

General steps to run tasks in multilingual ASR task are as follows:
- Step1: Following [downloading guide](https://github.com/espnet/espnet/blob/master/egs2/ml_superb/asr1/README.md#data-downloadsetup) to prepare the data
- Step2: Adding the training configurations for the desired model at `conf/tuning` (check examples `conf/tuning/train_asr_s3prl_{10min, 1h}.yaml` and `conf/tuning/train_asr_fbank_{10min, 1h}.yaml`) **Note: only the frontend/learning rate can be changed for the benchmark.**
- Step3: Training the 10min/1h model by calling
```
./run_multi.sh --asr_config <your_training_config> --duration {10min, 1h}
```


### LID

General steps to run tasks in LID trask are as follows:
- Step1: Following [downloading guide](https://github.com/espnet/espnet/blob/master/egs2/ml_superb/asr1/README.md#data-downloadsetup) to prepare the data
- Step2: Adding the training configurations for the desired model at `conf/tuning` (check examples `conf/tuning/train_asr_s3prl_{10min, 1h}.yaml` and `conf/tuning/train_asr_fbank_{10min, 1h}.yaml`) **Note: only the frontend/learning rate can be changed for the benchmark.**
- Step3: Training the 10min/1h model by calling
```
./run_multi.sh --asr_config <your_training_config> --duration {10min, 1h} --only_lid true
```


### Multilingual ASR+LID

General steps to run tasks in LID trask are as follows:
- Step1: Following [downloading guide](https://github.com/espnet/espnet/blob/master/egs2/ml_superb/asr1/README.md#data-downloadsetup) to prepare the data
- Step2: Adding the training configurations for the desired model at `conf/tuning` (check examples `conf/tuning/train_asr_s3prl_{10min, 1h}.yaml` and `conf/tuning/train_asr_fbank_{10min, 1h}.yaml`) **Note: only the frontend/learning rate can be changed for the benchmark.**
- Step3: Training the 10min/1h model by calling
```
./run_multi.sh --asr_config <your_training_config> --duration {10min, 1h} --lid true
```

## Credits

We would like to thank the following resources:

1. S.-w. Yang et al., “SUPERB: Speech Processing Universal PERformance Benchmark,” in Proc. Interspeech, 2021, pp. 1194–1198
2. H.-S. Tsai et al., “SUPERB-SG: Enhanced speech processing universal performance benchmark for semantic and generative capabilities,” in Proc. ACL, 2022, pp. 8479–8492.
3. V. Pratap et al., “MLS: A large-scale multilingual dataset for speech research,” Proc. Interspeech 2020, pp. 2757–2761, 2020.
4. R. Ardila et al., “Common voice: A massively-multilingual speech corpus,” in Proc. LREC, 2020, pp. 4218–4222
5. K. MacLean, “Voxforge,” Ken MacLean.[Online]. Available: http://www. voxforge. org/home.[Accessed by 2022], 2018.
6. C. Wang et al., “Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation,” in Proc. ACL, 2021, pp. 993–1003.
7. K. Sodimana et al., “A step-by-step process for building TTS voices using open source data and framework for Bangla, Javanese, Khmer, Nepali, Sinhala, and Sundanese,” in Proc. SLTU, 2018, pp. 66–70.
8. O. Kjartansson et al., “Open-source high quality speech datasets for Basque, Catalan and Galician,” in Proc. SLTU, 2020, pp. 21–27
9. F. He et al., “Open-source multi-speaker speech corpora for building Gujarati, Kannada, Malayalam, Marathi, Tamil and Telugu speech synthesis systems,” in Proc. LREC, 2020, pp. 6494–6503.
10. G. Rehm and H. Uszkoreit, “Language technology support for Norwegian,” in The Norwegian Language in the Digital Age: Bokmalsversjon, 2012, pp. 52–70.
11. A. Conneau et al., “FLEURS: Few-shot learning evaluation of universal representations of speech,” in Proc. SLT, 2023, pp. 798–805
12. E. Barnard et al., “The NCHLT speech corpus of the south African languages,” 2014.
13. T. Baumann, A. K ̈ohn, and F. Hennig, “The spoken wikipedia corpus collection: Harvesting, alignment and an application to hyperlistening,” LREC, vol. 53, pp. 303–329, 2019.
14. J. Shi et al., “Leveraging end-to-end ASR for endangered language documentation: An empirical study on Yol ́oxochitl Mixtec,” in Proc. ACL, 2021, pp. 1134–1145.
15. J. Shi et al., “Highland Puebla Nahuatl speech translation corpus for endangered language documentation,” in Proc. AmericaNLP, 2021, pp. 53–63.
16. I. Solak, “M-AILAB speech dataset,” Imdat Solak.[Online]. Available: https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/.[Accessed by 2022], 2018.
17. D. A. Braude et al., “All together now: The living audio dataset.,” in Proc. Interspeech, 2019, pp. 1521–1525.
18. N. J. De Vries et al., “A smartphone-based ASR data collection tool for under-resourced languages,” Speech communication, vol. 56, pp. 119–131, 2014.
