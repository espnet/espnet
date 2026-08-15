# MyST RECIPE

This is the recipe of the children speech recognition model with [MyST dataset](https://catalog.ldc.upenn.edu/LDC2021S05).

Before running the recipe, please download from https://catalog.ldc.upenn.edu/LDC2021S05.
Then, edit 'MYST' in `db.sh` and locate unzipped dataset as follows:

```bash
$ vim db.sh
MYST=/path/to/myst

$ tree -L 2 /path/to/myst
/path/to/myst
└── myst_child_conv_speech
    ├── data
    ├── docs
    └── index.html
```


# PEFT Fine-tuning

This recipe also provides example configurations for parameter-efficient fine-tuning (PEFT) of OWSM on the MyST corpus.

The PEFT examples fine-tune selected linear layers of the pretrained model using low-rank adaptation variants, while keeping the main pretrained model parameters frozen. The backend is selected directly in the YAML configuration.

## Supported LoRA backends

Depending on the selected configuration, the recipe may use one of the following LoRA variants:

- **LoRA**: https://arxiv.org/pdf/2106.09685
- **SSVD**: https://arxiv.org/pdf/2509.02830
- **DoRA**: https://arxiv.org/pdf/2402.09353
- **PiSSA**: https://arxiv.org/pdf/2404.02948
- **SVFT**: https://arxiv.org/pdf/2405.19597

These methods are configured through the YAML files using `adapter_type`, with supported values including `lora`, `ssvd``, ``dora`, `pissa`, and `svft`.

## Example configuration

An example SSVD configuration is provided in:
```bash
conf/tuning/peft_tuning_owsm_ssvd.yaml
```

A LoRA configuration can be enabled by setting `use_adapter: true` and selecting the backend through `adapter_conf.adapter_type.`

```yaml
use_adapter: true
adapter: lora
save_strategy: all
adapter_conf:
    rank: 0
    alpha: 0
    dropout_rate: 0.0
    adapter_type: ssvd
    rotation_ratio: 0.4
    target_modules: [attn.linear_q, attn.linear_v]

```

Here, `adapter_type: ssvd` selects the SSVD backend. To use another LoRA backend, change `adapter_type`, for example:

```yaml
adapter_type: dora
```

# RESULTS

## exp/s2t_peft_tuning_owsm_ssvd

Model: https://huggingface.co/wangpuupup/myst_peft_tuning_owsm_ssvd

## Environments
- date: `Wed May  6 00:29:18 EDT 2026`
- python version: `3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]`
- espnet version: `espnet 202503`
- pytorch version: `pytorch 2.4.0`
- Git hash: `1efdaa835178b0ce5034904e29f89f8fc7e0a358`
  - Commit date: `Thu May 22 12:09:45 2025 -0400`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_15epoch/test_filter|10328|184823|89.2|7.2|3.6|2.9|13.8|65.9|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_15epoch/test_filter|10328|927685|93.9|2.0|4.1|3.0|9.2|65.9|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_15epoch/test_filter|10328|673551|92.4|2.9|4.7|2.7|10.3|65.9|


## exp/asr_asr_train_asr_wavlm_transformer_raw_en_bpe5000_sp_bs16000000

Model: https://huggingface.co/espnet/myst_wavlm_aed_transformer

## Environments
- date: `Mon Nov 25 21:12:07 CST 2024`
- python version: `3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 19:46:43) [GCC 11.2.0]`
- espnet version: `espnet 202409`
- pytorch version: `pytorch 2.4.0`
- Git hash: `6b5c6230a794aa4a5df872be69e417a3fbfe821b`
  - Commit date: `Sun Nov 24 23:13:48 2024 -0600`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|13180|202306|88.4|7.6|4.0|3.4|15.0|61.9|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|13180|1016043|93.2|2.1|4.7|3.6|10.4|61.9|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|13180|228240|86.4|6.7|6.8|4.0|17.6|61.9|


# References
[1] Pradhan, Sameer, Ronald Cole, and Wayne Ward. "My Science Tutor (MyST)–a Large Corpus of Children’s Conversational Speech." Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024). 2024.

[2] Wang, Pu, Shinji Watanabe, and Hugo Van Hamme. "SSVD: Structured SVD for Parameter-Efficient Fine-Tuning and Benchmarking under Domain Shift in ASR," 2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), Honolulu, HI, USA, 2025, pp. 1-7, doi: 10.1109/ASRU65441.2025.11434624.

[3] Wang, Pu, Shinji Watanabe, and Hugo Van Hamme. "SSVD-O: Parameter-Efficient Fine-Tuning with Structured SVD for Speech Recognition." ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2026, pp. 16632-16636, doi: 10.1109/ICASSP55912.2026.11462142.
