# ACE-Opencpop SVS recipe

Port of [`egs2/acesinger/svs1`](../../../egs2/acesinger/svs1) to ESPnet3.
ACE-Opencpop ([Shi et al., 2024](https://arxiv.org/abs/2401.17619)) is the
Opencpop song set re-sung by 30 ACE Studio singers: 105,000 segments at
44.1 kHz with phoneme alignments and music scores, CC BY-NC 4.0.

The corpus is downloaded automatically from
[`espnet/ace-opencpop-segments`](https://huggingface.co/datasets/espnet/ace-opencpop-segments)
on the Hugging Face Hub (41 GB), which holds the same segments, phoneme
alignments and scores that the egs2 recipe builds from the raw archives.
As in egs2, song 2026 is left out of training: 99,190 / 50 / 5,400 segments
for train / valid / test.

## Quick start

```bash
# 1) Download the corpus, write wav files + manifests (about 30 min)
python run.py --stages create_dataset --training_config conf/tuning/train_visinger2.yaml

# 2) Filter utterances by duration and build the phoneme token list
python run.py --stages remove_long_short create_token_list \
    --training_config conf/tuning/train_visinger2.yaml

# 3) Collect feature statistics; fbank and pitch are dumped for training
python run.py --stages collect_stats --training_config conf/tuning/train_visinger2.yaml

# 4) Train VISinger 2 on 4 GPUs
python run.py --stages train --training_config conf/tuning/train_visinger2.yaml

# 5) Synthesize the test set from its music scores and score it
python run.py --stages infer measure \
    --training_config conf/tuning/train_visinger2.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`collect_stats` over 99k utterances is slow with a single local worker. Set
`parallel.env` to a cluster backend and supply your scheduler's `options` in
the training config.

## Results

| dataset | MCD | F0 RMSE (Hz) | F0 corr | SingMOS |
| --- | --- | --- | --- | --- |
| test | 5.76 | 59.3 | 0.679 | 4.38 |

Decoded with `conf/inference.yaml` from the `valid.generator.loss.ave_5best`
average after 100 of the configured 500 epochs on 4 GPUs, and scored with
VERSA (`conf/metrics.yaml`). SingMOS of the reference recordings is 4.37.

## Pretrained Models

- [`jjiang4/acesinger_svs_train_visinger2`](https://huggingface.co/jjiang4/acesinger_svs_train_visinger2):
  the model above, packed with `pack_model` / `upload_model`
  (`conf/publication.yaml`).

```python
# From this directory after `. ./path.sh` and `create_dataset`.
from dataset import Dataset
from espnet3.publication import InferenceModel

model = InferenceModel.from_pretrained(
    "jjiang4/acesinger_svs_train_visinger2", trust_user_code=True
)
# One item of the recipe dataset built with `inference: true`: the music
# score under "text" and the singer id "sids".
sample = Dataset(split="test", inference=True)[0]
wav = model(sample)["wav"]  # 44.1 kHz
```

## Differences from the egs2 recipe

These are deliberate; each is marked `[DEVIATION]` in the configs.

- **Data comes from the Hub.** egs2 segments the raw ACE Studio recordings
  itself; this recipe downloads the published segments and resamples them
  from 48 kHz to 44.1 kHz. The content is the same. The splits are called
  `train` / `valid` / `test` instead of `tr_no_dev` / `dev` / `eval`.
- **Larger batches.** egs2 spreads its batch of 8 over 4 GPUs; here each GPU
  gets 8, so one update sees 32 utterances. The learning rate is the same.
- **Model selection.** egs2 decodes the last checkpoint; this recipe averages
  the 5 checkpoints with the best validation generator loss.
- **Only the test set is scored.** The validation set has 50 utterances.
- **Metrics.** VERSA's `mcd_f0` (MCD, F0 RMSE, F0 correlation) and SingMOS.
  egs2 also reports semitone accuracy and V/UV error.
