# LibriSpeech 100h ASR recipe

The recipe reads the existing corpus at `download/LibriSpeech` by default. To
use a shared copy, set `LIBRISPEECH` to its root directory. The builder creates
Hugging Face Dataset indexes containing only each utterance's audio path and
transcript under `data/huggingface/librispeech_asr/`; it does not copy or
re-encode the audio.

From the repository root, install the training dependencies. Then, from this
recipe directory, build the train, validation, and test indexes:

```bash
# From the repository root
pip install -e '.[train]'

# From egs3/librispeech_100/esp2_asr
export LIBRISPEECH=/path/to/LibriSpeech
python run.py --stages create_dataset \
    --training_config conf/tuning/training_e_branchformer.yaml
```

Training and inference load their split indexes from `data/` and read audio
from the shared corpus path.

## Prepare training

Train the BPE tokenizer, then collect feature statistics using four Dask
workers submitted to the CPU Slurm partition:

```bash
export LIBRISPEECH=/path/to/LibriSpeech
python run.py --stages train_tokenizer collect_stats \
    --training_config conf/tuning/training_e_branchformer.yaml
```

The Slurm worker settings are in the training config's `parallel` section.

## Quick start

```bash
# 1) Train the default E-Branchformer model
python run.py --stages train \
    --training_config conf/tuning/training_e_branchformer.yaml

# 2) Decode
python run.py --stages infer \
    --training_config conf/tuning/training_e_branchformer.yaml \
    --inference_config conf/inference.yaml

# 3) Score
python run.py --stages measure \
    --training_config conf/tuning/training_e_branchformer.yaml \
    --metrics_config conf/metrics.yaml
```

## Pretrained Models

- [`conf/tuning/training_e_branchformer.yaml`](https://huggingface.co/ms180/librispeech_100h_e_branchformer)
