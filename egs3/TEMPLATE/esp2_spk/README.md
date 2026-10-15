# ESPnet3 speaker verification recipe

The template holds the default configs of the `esp2_spk` system
(`espnet3/systems/esp2_spk`). A recipe provides the dataset module and the
model configuration, and its `run.py` is only:

```python
from espnet3.systems.esp2_spk.system import SpeakerSystem
from espnet3.systems.launch import launch

if __name__ == "__main__":
    launch(SpeakerSystem)
```

`SpeakerSystem` declares the stages `create_dataset`, `train`, `infer`,
`measure`, `pack_model` and `upload_model`: training on raw waveform crops
needs neither a tokenizer nor `collect_stats`.

Training is a closed-set classification over the training speakers, while
validation and evaluation score verification trials, so `best_model_criterion`
tracks `valid/eer` rather than a loss.

## Quick start

```bash
# 1) Prepare Kaldi-style manifests and trial lists
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Train
python run.py --stages train --training_config conf/training.yaml

# 3) Score the trial lists
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 4) Compute EER and minDCF
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

## Dataset contract

`dataset/__init__.py` must expose `Dataset` and `DatasetBuilder`. The dataset
is an `espnet3.components.data.base_dataset.BaseDataset` and declares its
`fields`; it returns two kinds of items, selected by its constructor
arguments:

| Split       | Keys                                | Meaning                       |
|---          |---                                  |---                            |
| train       | `speech`, `spk_labels`              | one utterance, speaker string |
| valid, test | `speech`, `speech2`, `spk_labels`   | trial pair, `1` if target     |

During training, `SpkPreprocessor` converts both forms into fixed-length crops
and maps the training speaker strings to integer labels using the `spk2utt`
file written by `create_dataset`. The `infer` stage uses no preprocessor:
`espnet3.systems.esp2_spk.inference.Inference` takes a trial's two whole
utterances, cuts the same evaluation crops itself, and returns `{"score": ...}`,
so a published model scores raw recordings exactly as `infer` does.
