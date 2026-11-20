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



# PEFT Fine-tuning of OWSM

This recipe also provides example configurations for parameter-efficient
fine-tuning (PEFT) of [OWSM v3.1](https://huggingface.co/espnet/owsm_v3.1_ebf)
on the MyST corpus.

The PEFT examples fine-tune selected linear layers of the pretrained model
using low-rank adaptation variants, while the main pretrained parameters stay
frozen. The backend is selected directly in the YAML configuration via
`adapter_conf.adapter_type`.

## Supported adapter backends

- **LoRA**: https://arxiv.org/abs/2106.09685
- **DoRA**: https://arxiv.org/abs/2402.09353
- **PiSSA**: https://arxiv.org/abs/2404.02948
- **SVFT**: https://arxiv.org/abs/2405.19597
- **SSVD**: https://arxiv.org/abs/2509.02830

Example configurations are provided in `conf/tuning/peft_tuning_owsm_*.yaml`
(`lora`, `dora`, `pissa`, `svft`, `ssvd`). A backend is enabled with:

```yaml
use_adapter: true
adapter: lora
adapter_conf:
    rank: 32
    alpha: 64
    dropout_rate: 0.0
    adapter_type: lora   # or: dora, pissa, svft, ssvd
    target_modules: [attn.linear_q, attn.linear_v]
```

**SSVD rotation map.** `rotation_map: cayley` applies the exact Cayley
transform `(I - S)^{-1}(I + S)`, which is strictly orthogonal.
`rotation_map: linear` (default, the setting used in the paper) evaluates the
Cayley transform via a truncated Neumann series, keeping only the first-order
term `I + 2S`; this introduces a small orthogonality error but is much faster.
Both share the same parameters, so checkpoints are interchangeable.
Experiments on MyST / OWSM v3.1 show the gap between the two mappings is
negligible (see [2], Section V-C, Table VI).

Since OWSM is a speech-to-text (s2t) model, these configurations are trained
with the s2t recipe (`s2t.sh`, as in `egs2/owsm_v3.1/s2t1`), pointing
`--s2t_config` at one of the yamls above, with `init_param` /
`normalize_conf.stats_file` referring to the downloaded
[espnet/owsm_v3.1_ebf](https://huggingface.co/espnet/owsm_v3.1_ebf) checkpoint
directory (see the comments inside the yamls).

# RESULTS

## exp/s2t_peft_tuning_owsm_ssvd (current espnet master)

Configuration: `conf/tuning/peft_tuning_owsm_ssvd.yaml`, 15 epochs, 1 GPU.

## Environments (current espnet master)
- date: `Mon Aug 31 22:24:49 CEST 2026`
- python version: `3.10.21 | packaged by conda-forge`
- espnet version: `espnet 202604`
- pytorch version: `pytorch 2.7.1`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_15epoch/test_filter|10328|184823|89.2|7.2|3.6|2.8|13.6|65.9|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_15epoch/test_filter|10328|927685|93.9|2.0|4.0|2.9|9.0|65.9|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_15epoch/test_filter|10328|673551|92.4|2.9|4.6|2.7|10.3|65.9|

## exp/s2t_peft_tuning_owsm_ssvd (espnet 202503 reference run)

Model: https://huggingface.co/wangpuupup/myst_peft_tuning_owsm_ssvd

## Environments (202503 reference run)
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

[1] Pradhan, Sameer, Ronald Cole, and Wayne Ward. "My Science Tutor (MyST)--a Large Corpus of Children's Conversational Speech." Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024). 2024.

[2] Wang, Pu, Shinji Watanabe, and Hugo Van Hamme. "SSVD: Structured SVD for Parameter-Efficient Fine-Tuning and Benchmarking under Domain Shift in ASR," ASRU 2025, doi: 10.1109/ASRU65441.2025.11434624. https://arxiv.org/abs/2509.02830

[3] Wang, Pu, Shinji Watanabe, and Hugo Van Hamme. "SSVD-O: Parameter-Efficient Fine-Tuning with Structured SVD for Speech Recognition." ICASSP 2026, doi: 10.1109/ICASSP55912.2026.11462142.
