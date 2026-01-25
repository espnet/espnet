---
title: ESPnet3 Train Dataset
---

# ESPnet3 Train Dataset

ESPnet3 expects a `DataOrganizer`-based dataset config for training. The typical
flow is:

1. Write a dataset class (any backend is fine).
2. Configure it under `dataset` in `train.yaml`.
3. Instantiate via Hydra and iterate train/valid/test.

## 1) Write your dataset class

You can build datasets using whatever backend you prefer, such as
[Hugging Face Datasets](https://huggingface.co/docs/datasets/en/index),
[Lhotse](https://github.com/lhotse-speech/lhotse/tree/master), or
[Arkive](https://github.com/wanchichen/arkive).

Here is a minimal ASR-style example. The dataset receives `manifest_path` from
the config, loads entries once, and indexes into them in `__getitem__`.

```python
class MiniAN4Dataset:
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)
        self._entries = _read_manifest(self.manifest_path)

    def __getitem__(self, idx):
        entry = self._entries[int(idx)]
        return {
            "speech": np.asarray(entry["array"], dtype=np.float32),
            "text": entry["text"],
        }

    def __len__(self):
        # Example: return 100 if the manifest has 100 entries.
        return len(self._entries)
```

## 2) Configure it in train.yaml

Each list item maps to a `DatasetConfig` entry. `DataOrganizer` will:

- Combine `train` and `valid` lists into per-split datasets.
- Keep `test` as named datasets for inference/evaluation.

```yaml
dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  train:
    - name: train_nodev
      dataset:
        _target_: src.dataset.MiniAN4Dataset
        manifest_path: ${dataset_dir}/manifest/train_nodev.tsv
    - name: train_2
      dataset:
        _target_: src.dataset.MiniAN4Dataset
        manifest_path: ${dataset_dir}/manifest/train_2.tsv
  valid:
    - name: train_dev
      dataset:
        _target_: src.dataset.MiniAN4Dataset
        manifest_path: ${dataset_dir}/manifest/train_dev.tsv
    - name: dev_2
      dataset:
        _target_: src.dataset.MiniAN4Dataset
        manifest_path: ${dataset_dir}/manifest/dev_2.tsv
  test:
    - name: test
      dataset:
        _target_: src.dataset.MiniAN4Dataset
        manifest_path: ${dataset_dir}/manifest/test.tsv
    - name: test_2
      dataset:
        _target_: src.dataset.MiniAN4Dataset
        manifest_path: ${dataset_dir}/manifest/test_2.tsv
  preprocessor:
    _target_: espnet2.train.preprocessor.CommonPreprocessor
    token_type: bpe
    token_list: ${tokenizer.save_path}/tokens.txt
    bpemodel: ${tokenizer.save_path}/bpe.model
```

Notes:

- `train` and `valid` must be both present or both omitted.
- `test` is optional and is typically used by `infer`/`measure`, not by `train`.
- `preprocessor` is used when you want to reuse ESPnet2 preprocessors. Choose
  from the implementations in
  [espnet2/train/preprocessor.py](https://github.com/espnet/espnet/blob/72a3db47bb26bf3ac2d43b055b517572bce67e38/espnet2/train/preprocessor.py)
  or implement your own with the same input/output contract.

## 3) Instantiate and use in Python

This is the basic way to use `DataOrganizer` in Python: instantiate the config,
iterate each split, and access named test sets when needed.

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

cfg = OmegaConf.load("conf/train.yaml")
organizer = instantiate(cfg.dataset)
```

### Train split

Iterate the training data. If you listed multiple train datasets in
the config, `DataOrganizer` joins them into one long list, so you can treat it
as a single dataset. In this example, each train dataset has 100 items, so
`len(organizer.train)` becomes 200.

```python
for sample in organizer.train:
    print(sample)
    break
# Example:
# {"speech": np.ndarray(...), "text": "SOME TEXT"}
len(organizer.train) == 200
```

### Valid split

Same idea as train, but for validation data. With two 100-item valid datasets,
`len(organizer.valid)` becomes 200.

```python
for sample in organizer.valid:
    pass
len(organizer.valid) == 200
```


### Test sets

Test sets are kept separately by name, so you can pick a specific test set or
loop over all of them.

```python
for name, test_set in organizer.test.items():
    for sample in test_set:
        pass

for sample in organizer.test["test"]:
    pass
for sample in organizer.test["test_2"]:
    pass
len(organizer.test["test"]) == 100
len(organizer.test["test_2"]) == 100
```

### UID + sample mode

If you use ESPnet's collate function, ESPnet3 automatically switches to
`(uid, sample)` pairs for train/valid.

```python
organizer.train.use_espnet_collator = True
organizer.valid.use_espnet_collator = True

for uid, sample in organizer.train:
    pass
```
