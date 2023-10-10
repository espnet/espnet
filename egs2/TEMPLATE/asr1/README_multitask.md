# ESPnet2 ASR1 Multi-tasking Recipe TEMPLATE

This is a template of ASR1 Multi-tasking recipe for ESPnet2.
This README provides comprehensive instructions on how to enhance ASR1 for prompt-based multi-task learning.

## Table of Contents

* [ESPnet2 ASR1 Multi-tasking Recipe TEMPLATE](#espnet2-asr2-recipe-template)
  * [Table of Contents](#table-of-contents)
  * [Recipe flow](#recipe-flow)
    * [1\. Data preparation](#1-data-preparation)
    * [2\. Speed perturbation](#2-speed-perturbation)
    * [3\. Generate dump folder](#3-generate-dump-folder)
    * [4\. Removal of long / short data](#4-removal-of-long--short-data)
    * [5\. Input / Output Token list generation](#5-input--output-token-list-generation)
    * [6\. LM statistics collection](#6-lm-statistics-collection)
    * [7\. LM training](#7-lm-training)
    * [8\. LM perplexity](#8-lm-perplexity)
    * [9\. Ngram-LM training](#9-n-gram-lm-training)
    * [10\. ASR statistics collection](#10-asr-statistics-collection)
    * [11\. ASR training](#11-asr-training)
    * [12\. ASR inference](#12-asr-inference)
    * [13\. ASR scoring](#13-asr-scoring)
    * [14\-16\. (Optional) Pack results for upload](#14-16-optional-pack-results-for-upload)
  * [How to run](#how-to-run)
    * [SLU Multi-task training](#slu-multi-task-training)
  * [Related works](#related-works)

## Recipe flow

ASR1 recipe consists of 13 stages.

### 1. Data preparation

Data preparation stage.

#### ESPnet format:

It calls `local/data.sh` to creates Kaldi-style data directories in `data/` for training, validation, and evaluation sets. In addition to the files in the `asr1` recipe, it generates an additional file called `prompt` that specifies the task to be performed for the given utterance..

- `prompt` format
    ```
    uttidA <prompt>
    uttidB <prompt>
    ...
    ```

See also:
- [About Kaldi-style data directory](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE#about-kaldi-style-data-directory)

### 2. Speed perturbation

Augment training data with speed perturbation. `data/${train_set}_spXX` would be generated (`XX` means the speed factor). This step is optional.

### 3. Generate dump folder

Dumping stage.
This stage move necessary files for training from `data` folder to `dump` folder.

### 4. Removal of long / short data

This stage is the same as that in ASR recipes. At this stage, the dump directories for all datasets on which multi-tasking is to be performed are merged by simple concatenation.

### 5. Input / Output Token list generation

Token list (BPE / Char / etc) generation for both input and targets. Additionally, for Whisper tokenization, you have the option to incorporate special tokens into the Whisper vocabulary using the `--nlsyms_txt` flag. If you are utilizing task specifiers for prompt-based multi-tasking, similar to the original Whisper formulation, it is necessary to include these task specifiers in the Whisper vocabulary.

### 6. LM statistics collection

Neural-network (NN) based Language model (LM) is optional for ASR task. You can skip stage 5-8 by setting `--use_lm false`.
Statistics calculation stage.
It collects the shape information of LM texts and calculates statistics for LM training.

### 7. LM training

NN-based LM model training stage.
You can change the training setting via `--lm_config` and `--lm_args` options.

See also:
- [Supported models](#supported-models).
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)

### 8. LM perplexity

NN-based LM evaluation stage. Perplexity (PPL) is computed against the trained model

See also:
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)

### 9. N-gram LM training

N-gram-based LM model training stage.


### 10. ASR statistics collection

Statistics calculation stage.
It collects the shape information of input and output texts for ASR training.
#### Prompt based multi-tasking

- Instructions:
  1. To enable prompt-based multi-task learning across multiple tasks in English, ensure that `--use_prompt` is set to True. By default, this setting replaces the task specifier in the Whisper formulation with the one specified in the prompt file to perform multi-task learning across multiple tasks in English.  Please refer to stage 5 for instructions on adding task specifiers to the Whisper vocabulary.
  2. If you want to perform prompt-based multi-task learning across multiple tasks in multiple languages, additionally, set `--use_lang_prompt` to true. This step replaces both the language and task specifiers in the Whisper formulation with those specified in the prompt file and can also introduce a new dataset specifier. Please ensure that task, dataset, and language specifiers are all included in the Whisper vocabulary for this option to work.
  3. (Optional) To use natural language phrases for prompt-based multi-tasking, set `--use_nlp_prompt` to true. In this case, you do not need to make any modifications to the Whisper vocabulary.

### 11. ASR training

ASR model training stage.
You can change the training setting via `--asr_config` and `--asr_args` options. You need to follow similar steps as described in stage 10 to perform prompt based multi-task learning.

See also:
- [Supported models](#supported-models).
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)

### 12. ASR inference

ASR inference stage.

#### Prompt based multi-tasking

- Instructions:
  1. If you have incorporated any special tokens into the Whisper vocabulary, make sure to specify the file containing these special tokens as `prompt_token_file` in decoder config.
  2. If you are utilizing task, language, and dataset specifiers, please specify these specifiers as `lang_prompt_token` in decoder config.
  3. If you are employing a natural language phrase as a prompt, specify the phrase as `nlp_prompt_token` in decoder config.
  4. To perform language identification and voice activity detection, we follow the Whisper's pre-training setupwhere we predict ``language id`` and ``no speech`` tags immediately after the start of the transcript tag.  Hence for these tasks, set ``lid prompt`` to true.

### 13. ASR scoring

ASR scoring stage: error rates (char / word / token) are computed.

### 14-16. (Optional) Pack results for upload

Packing stage.
It packs the trained model files and uploads to [Zenodo](https://zenodo.org/) (Zenodo upload will be deprecated).
If you want to run this stage, you need to register your account in zenodo.

See also:
- [ESPnet Model Zoo](https://github.com/espnet/espnet_model_zoo)

#### Stage 16-18: Upload model

Upload the trained model to Hugging Face for sharing. Additional information at [Docs](https://espnet.github.io/espnet/espnet2_tutorial.html#packing-and-sharing-your-trained-model).

## How to run

### SLU-Multi-task-training
Here, we show the procedure to run multi-tasking learning across 14 speech classification tasks.


Create a dump directory using the following recipes: . ``asvspoof, speechcommands, grabo, lt_speech_commands, arabic_sc, fsc, voxforge/lid1, iemocap, accentdb, mustard, mustard_plus_plus, voxceleb1, freesound and esc50``. You can do this by running the following command in each of these recipes:
```sh
$ ./run.sh --stop_stage 4
```
Note: Download all the dataset zip files first before creating dump directory. Please refer to ``https://github.com/ga642381/SpeechPrompt-v2/blob/main/docs/dataset.md`` to download all datasets.


Move to the `egs2/uslu14/asr1` recipe directory. Generate the `prompt` file by running
```sh
$ python local/create_*_prompt.py
```


Concatenate ``wav.scp, prompt, text, utt2spk, spk2utt, utt2num_samples`` from all train and valid dump folders in each of the dump directories and create two new directories, ``dump/raw/train_combined`` and ``dump/raw/valid`` to contain the combined data. Start training using:
```sh
$ ./run.sh --stage 5 --stop_stage 11
```


Run decoding for each of the datasets, i.e., ``test_<dataset>``, with the specified inference_config, e.g., ``conf/decode_asr_<task>.yaml``, using the following command:
```sh
$ ./run.sh --stage 12 --stop_stage 12  --inference_config conf/decode_asr_<task>.yaml --test_sets test_<dataset>
```


For some tasks, you need to clean prediction files using ``python local/clean_emotion_pred.py``, ``python local/check_lid_results.py``, ``python local/check_vad_results.py``. To get accuracy, run
```sh
$ ./run.sh --stage 13 --stop_stage 13  --inference_config conf/decode_asr_<task>.yaml --test_sets test_<dataset>
```
For tasks where you need to compute f1 or weighted_f1, run ``python local/compute_f1.py`` and ``python local/compute_weighted_f1.py``.


## Related works
```

@misc{arora2023universlu,
      title={UniverSLU: Universal Spoken Language Understanding for Diverse Classification and Sequence Generation Tasks with a Single Network},
      author={Siddhant Arora and Hayato Futami and Jee-weon Jung and Yifan Peng and Roshan Sharma and Yosuke Kashiwagi and Emiru Tsunoo and Shinji Watanabe},
      year={2023},
      eprint={2310.02973},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@InProceedings{pmlr-v202-radford23a,
  title = 	 {Robust Speech Recognition via Large-Scale Weak Supervision},
  author =       {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and Mcleavey, Christine and Sutskever, Ilya},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {28492--28518},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
}
```
