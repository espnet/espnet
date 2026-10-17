# README

[Paper](https://arxiv.org/pdf/2508.17148)

This geolocation-aware language identification (LID) model is developed using the [ESPnet](https://github.com/espnet/espnet/) toolkit. It integrates the powerful pretrained [MMS-1B](https://huggingface.co/facebook/mms-1b) as the encoder and employs [ECAPA-TDNN](https://arxiv.org/pdf/2005.07143) as the embedding extractor to achieve robust spoken language identification.

The main innovations of this model are:
1. Incorporating geolocation prediction as an auxiliary task during training.
2. Conditioning the intermediate representations of the self-supervised learning (SSL) encoder on intermediate-layer information.
This geolocation-aware strategy greatly improves robustness, especially for dialects and accented variations.

For further details on the geolocation-aware LID methodology, please refer to our paper: *Geolocation-Aware Robust Spoken Language Identification* ([arXiv](https://arxiv.org/pdf/2508.17148)).

## Usage

### Prerequisites

First, ensure you have ESPnet installed. If not, follow the [ESPnet installation instructions](https://espnet.github.io/espnet/installation.html).

The recipe uses the official `transformers` package. Neither a custom Transformers
installation nor S3PRL is required for geolocation conditioning. Training also uses
`lang2vec`; dataset preparation requires `datasets`, `pandas`, and `pycountry`.

```bash
pip install 'transformers>=4.51.3' lang2vec datasets pandas pycountry
```

The checkpoint-local models have been tested with official Transformers 4.51.3
and 5.14.1. The conditioning implementation is in
`espnet2/lid/huggingface/modeling_wav2vec2_condition.py`.

### Inference with Transformers

Each model linked below includes `config.json`, `model.safetensors`, a feature
extractor configuration, and the custom `modeling_*.py` / `configuration_geolid.py`
files. These files load through the standard AutoClass API:

```python
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

repo = "espnet/geolid_vl107only_shared_trainable"
extractor = AutoFeatureExtractor.from_pretrained(repo)
model = AutoModelForAudioClassification.from_pretrained(
    repo, trust_remote_code=True
).eval()
audio, sample_rate = sf.read("speech.wav", dtype="float32")
assert sample_rate == 16000 and audio.ndim == 1  # mono, at least 50 ms
inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
with torch.inference_mode():
    outputs = model(**inputs)
label = outputs.logits.argmax(-1).item()
print(model.config.id2label[label])
```

`trust_remote_code=True` loads the modeling files shipped with the checkpoint.
For reproducible deployments, pass a reviewed commit hash as `revision` to both
calls. This inference path needs only PyTorch and Transformers (plus an audio
reader); ESPnet and S3PRL are not required. The outputs contain classification
scores, language embeddings, and intermediate geolocation predictions.

The released models' language IDs were recovered from their original model-card
language lists and the recipe's sorted `lang2utt` construction. `conversion.json`
records this provenance. The original ESPnet `.pth` files are retained.

Published checkpoints set `legacy_inference=true`, preserving the pooling variance
floor and padded mean-normalization behavior used in 2025. Old ESPnet configs
without this field default to that behavior. New training configs explicitly use
`legacy_inference: false` to follow current ESPnet. The results below are the
original published evaluations; no models were retrained for this migration.

### Export an ESPnet checkpoint

From this recipe directory, with ESPnet installed:

```bash
python local/export_huggingface.py \
    --checkpoint /path/to/valid.accuracy.best.pth \
    --training-config /path/to/config.yaml \
    --lang2utt /path/to/original/training/lang2utt \
    --output-dir /path/to/new-hf-checkpoint
```

The exporter checks the architecture, loads weights strictly, and writes a
self-contained safetensors checkpoint. It supports the five published
configurations, including the early independent/frozen parameter names. It does
not retrain or quantize weights. If `--lang2utt` is omitted, labels remain numeric
`LABEL_i` values and `label_mapping_verified` is false; do not substitute the
labels of a test set. Use an empty output directory.

### ESPnet training and evaluation

**Step 1: Navigate to project directory**
```bash
cd espnet/egs2/geolid/lid1
```

If you would like to train the model or run inference using our prepared datasets (instead of your own), please first follow the [data preparation instructions](local/README.md) to set up the required data.

The standard entry point is `./run.sh`, using `conf/train.yaml` (VoxLingua107-only,
shared trainable conditioning). Variants live in `conf/tuning/`; the historical
configuration paths remain as symlinks for existing experiments. Data preparation
is described in `local/README.md`; begin at stage 4 after preparing the dumps.

**Option 1: Train with Combined Dataset**
```bash
# Uses all available datasets for training
./run_combined.sh --stage 4 --stop_stage 8
```

**Option 2: Train with VoxLingua107 Only**
```bash
# Four different configurations available
# Check run_voxlingua107_only.sh for configuration details
./run_voxlingua107_only.sh --stage 4 --stop_stage 8
```

We also provide pre-trained models for immediate use:

```bash
# Download the exp_combined to egs2/geolid/lid1
hf download espnet/geolid_vl107only_shared_trainable --local-dir . --include "exp_voxlingua107_only/*"

./run_voxlingua107_only.sh --skip_data_prep false --skip_train true --lid_config conf/voxlingua107_only/mms_ecapa_upcon_32_44_it0.4_shared_trainable.yaml --stage 6 --stop_stage 8

```

To run inference with checkpoints trained on the combined training set, please use the script `run_combined.sh`.

If you want to perform inference on your own prepared datasets, modify the test_sets field in the corresponding run script.

See the [Results](#results) section below for detailed performance metrics and Hugging Face model links.

## Train and Evaluation Datasets

Two training setups are provided:
1. VoxLingua107-Only: The training used only the VoxLingua107 dataset, comprising 6,628 hours of speech across 107 languages from YouTube.
2. Combined: The training utilized a combined dataset, merging five domain-specific corpora, resulting in 9,865 hours of speech data covering 157 languages.

Below is a summary table of the training and evaluation datasets used in this project:

| Dataset                                                      | Domain      | #Langs. Train/Test | Dialect | Training Setup (VL107-only) | Training Setup (Combined) |
| ------------------------------------------------------------ | ----------- | ------------------ | ------- | --------------------------- | ------------------------- |
| [VoxLingua107](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/) | YouTube     | 107/33             | No      | Seen                        | Seen                      |
| [Babel](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=31a13cefb42647e924e0d2778d341decc44c40e9) | Telephone   | 25/25              | No      | Unseen                      | Seen                      |
| [FLEURS](https://huggingface.co/datasets/google/xtreme_s)    | Read speech | 102/102            | No      | Unseen                      | Seen                      |
| [ML-SUPERB 2.0](https://huggingface.co/datasets/espnet/ml_superb_hf) | Mixed       | 137/(137, 8)       | Yes     | Unseen                      | Seen                      |
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) | Parliament  | 16/16              | No      | Unseen                      | Seen                      |


## Results

**Accuracy (%) on In-domain and Out-of-domain Test Sets**

| Config                    | 🤗 HF Repo | VoxLingua107 | Babel | FLEURS | ML-SUPERB2.0 Dev | ML-SUPERB2.0 Dialect | VoxPopuli | Macro Avg. |
| ------------------------- | ----------- | ------------ | ----- | ------ | ---------------- | -------------------- | --------- | ---------- |
| `conf/voxlingua107_only/mms_ecapa_upcon_32_44_it0.4_independent_frozen.yaml` | [espnet/geolid_vl107only_independent_frozen](https://huggingface.co/espnet/geolid_vl107only_independent_frozen) | 94.2         | 87.1  | 95.0   | 89.0             | 77.2                 | 90.4      | 88.8       |
| `conf/voxlingua107_only/mms_ecapa_upcon_32_44_it0.4_independent_trainable.yaml` | [espnet/geolid_vl107only_independent_trainable](https://huggingface.co/espnet/geolid_vl107only_independent_trainable) | 93.7         | 85.3  | 93.7   | 88.3             | 70.3                 | 86.5      | 86.3       |
| `conf/voxlingua107_only/mms_ecapa_upcon_32_44_it0.4_shared_frozen.yaml` | [espnet/geolid_vl107only_shared_frozen](https://huggingface.co/espnet/geolid_vl107only_shared_frozen) | 94.3         | 85.9  | 94.3   | 88.8             | 80.7                 | 89.2      | 88.8       |
| `conf/voxlingua107_only/mms_ecapa_upcon_32_44_it0.4_shared_trainable.yaml` | [espnet/geolid_vl107only_shared_trainable](https://huggingface.co/espnet/geolid_vl107only_shared_trainable) | 94.9         | 87.7  | 93.5   | 89.3             | 78.8                 | 89.5      | 88.9       |
| `conf/combined/mms_ecapa_upcon_32_44_it0.4_shared_trainable.yaml` | [espnet/geolid_combined_shared_trainable](https://huggingface.co/espnet/geolid_combined_shared_trainable) | 94.4         | 95.4  | 97.7   | 88.6             | 86.8                 | 99.0      | 93.7       |





## Citation

```BibTex
@inproceedings{wang2025geolid,
  author={Qingzheng Wang, Hye-jin Shim, Jiancheng Sun, and Shinji Watanabe},
  title={Geolocation-Aware Robust Spoken Language Identification},
  year={2025},
  booktitle={Proceedings of ASRU},
}

@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson Yalta and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={{ESPnet}: End-to-End Speech Processing Toolkit},
  year={2018},
  booktitle={Proceedings of Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1456}
}
```
