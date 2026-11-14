# LibriSpeech 100h ASR recipe

## Data Preparation

By default, the builder looks under `download/LibriSpeech`. After cloning the
recipe, either set `builder.dataset_path` in `dataset/config.yaml` to your
existing corpus location or set the `LIBRISPEECH` environment variable:

```bash
export LIBRISPEECH=/path/to/LibriSpeech
```

Both options accept the `LibriSpeech` directory or its parent. The configured
`dataset_path` is checked before the environment variable.

The corpus must already be downloaded and contain `train-clean-100`,
`dev-clean`, `dev-other`, `test-clean`, and `test-other`; the builder checks
these splits but does not download them. Run commands from
`egs3/librispeech_100/esp2_asr/`.

## Commands

### E-Branchformer

```bash
python run.py --stages create_dataset train_tokenizer collect_stats \
    --training_config conf/tuning/training_e_branchformer.yaml

python run.py --stages train \
    --training_config conf/tuning/training_e_branchformer.yaml

python run.py --stages infer measure \
    --training_config conf/tuning/training_e_branchformer.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

## Results

### E-Branchformer

- Training config: `conf/tuning/training_e_branchformer.yaml`
- Inference config: `conf/inference.yaml`
- Metrics config: `conf/metrics.yaml`

The [published model card](https://huggingface.co/ms180/librispeech_100h_e_branchformer)
reports the following results for an E-Branchformer with 38.47 million
parameters. These results were produced with an earlier version of the recipe;
rerunning the current config has not been verified to reproduce them.

| Dataset | WER | CER |
| --- | ---: | ---: |
| dev-clean | 6.36 | 2.55 |
| dev-other | 16.80 | 8.33 |
| test-clean | 6.41 | 2.49 |
| test-other | 17.12 | 8.31 |

The current inference config uses beam size 20, CTC weight 0.3, and the
average of the ten best `valid/acc` checkpoints. Consult the published model's
own configuration for the settings used for the reported results.

#### Pretrained model

- [E-Branchformer on Hugging Face](https://huggingface.co/ms180/librispeech_100h_e_branchformer)
