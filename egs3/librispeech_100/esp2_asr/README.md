# LibriSpeech 100h ASR recipe

## Data Preparation

Place the LibriSpeech corpus under `download/LibriSpeech`, or point
`LIBRISPEECH` to an existing LibriSpeech root. The recipe uses
`train-clean-100`, `dev-clean`, `dev-other`, `test-clean`, and `test-other`.
Run the commands below from `egs3/librispeech_100/esp2_asr/`.

```bash
export LIBRISPEECH=/path/to/LibriSpeech
```

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
