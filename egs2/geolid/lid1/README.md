# README

[Paper](https://arxiv.org/pdf/2508.17148)

This geolocation-aware language identification (LID) model is developed using the [ESPnet](https://github.com/espnet/espnet/) toolkit. It integrates the powerful pretrained [MMS-1B](https://huggingface.co/facebook/mms-1b) as the encoder and employs [ECAPA-TDNN](https://arxiv.org/pdf/2005.07143) as the embedding extractor to achieve robust spoken language identification.

The main innovations of this model are:
1. Incorporating geolocation prediction as an auxiliary task during training.
2. Conditioning the intermediate representations of the self-supervised learning (SSL) encoder on intermediate-layer information.
This geolocation-aware strategy greatly improves robustness, especially for dialects and accented variations.

For further details on the geolocation-aware LID methodology, please refer to our paper: *Geolocation-Aware Robust Spoken Language Identification* ([arXiv](https://arxiv.org/pdf/2508.17148)).

## Usage

Install ESPnet and the official SSL frontend dependencies:

```bash
pip install espnet-s3prl transformers
cd egs2/geolid/lid1
pip install -r local/requirements.txt
```

The recipe uses the stock `S3prlFrontend`, configured in `conf/train.yaml`:

```yaml
frontend: s3prl
frontend_conf:
  frontend_conf:
    upstream: hf_wav2vec2_custom
    path_or_url: facebook/mms-1b
  multilayer_feature: true
```

`ESPnetLIDUpstreamConditionModel` replaces encoder layers 32, 36, 40 and 44
(zero-based) with `Lang2VecConditionedLayer` wrappers. Each wrapper runs the
original SSL layer, pools its output, predicts a lang2vec vector and adds the
projected vector to the frame representations before the next SSL layer. The
model creates the projections, optionally shares them across layers, and collects
the intermediate predictions for auxiliary losses. `config.layerdrop` is set to
zero so these layers always execute. Attention and waveform preprocessing remain
in stock Transformers and S3PRL. There is no custom Transformers class, personal
fork, or checkpoint-local model code.

`cutoff_gradient_from_backbone: true` both detaches the projected residual before
it is added to the SSL hidden states and freezes the conditioning projection.
The SSL backbone and auxiliary prediction heads remain trainable.
`cutoff_gradient_before_condproj: true` instead detaches the prediction before
projection; this setting alone does not freeze the projection.

### Inference with a published ESPnet model

The five models below are normal ESPnet checkpoints (`config.yaml`, `model.pth`,
`meta.yaml` and `languages.txt`). After installing this recipe's ESPnet revision:

```python
import soundfile as sf
import torch
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.tasks.lid import LIDTask

files = ModelDownloader().download_and_unpack(
    "espnet/geolid_vl107only_shared_trainable"
)
model, train_args = LIDTask.build_model_from_file(
    files["train_config"], files["model_file"], device="cpu"
)
model.eval()
audio, sample_rate = sf.read("speech.wav", dtype="float32")
assert sample_rate == 16000 and audio.ndim == 1
speech = torch.from_numpy(audio)[None]
with torch.inference_mode():
    embeddings, prediction = model(
        speech, torch.tensor([len(audio)]), extract_embd=True
    )
with open(train_args.lang2utt) as stream:
    languages = [line.split()[0] for line in stream]
print(languages[prediction.item()])
```

The checkpoints were **converted, not retrained**. The conversion only renames
parameter keys and removes verified duplicate aliases; learned tensor values are
unchanged. Language order was recovered from the original model-card language
lists and the original recipe's sorted label construction. `languages.txt`
records that classifier order; it is not the original training utterance list.
Each repo includes `conversion.json` and a strict-loading/inference smoke-test
report in `validation.json`.

Runtime uses current ESPnet normalization and pooling. Legacy numerical behavior
is not emulated. The results below distinguish measured converted-model accuracy
from historical paper results. The models inherit **CC-BY-NC-4.0** from
[MMS-1B](https://huggingface.co/facebook/mms-1b); the earlier CC-BY-4.0 model-card
metadata has been corrected on all five repositories.

### Convert an original checkpoint

```bash
python local/convert_checkpoint.py \
    --checkpoint /path/to/original/valid.accuracy.best.pth \
    --config /path/to/original/config.yaml \
    --languages /path/to/languages.txt \
    --output-dir /path/to/converted
```

Use an empty output directory and one ISO-639-3 code per line in classifier-index
order. The script supports the original shared, independent and early frozen
projection names. Legacy key handling exists only in this conversion script.

### Training and recipe evaluation

Follow [local/README.md](local/README.md) to prepare the datasets and waveform
dumps. Missing lang2vec languages are reported as an error. Combined training
explicitly sets `lang2vec_exclude_languages: [tok]`: URIEL has no geographic
vector for Toki Pona. All 157 classification labels are retained, but `tok`
examples are masked out of both intermediate and downstream lang2vec losses.
The collated placeholder is never used as an auxiliary target. Other missing
languages still raise an error; a batch containing only excluded languages uses
classification loss alone.

The [combined startup check](results/combined_training_start.json) verified all
157 labels against lang2vec 1.1.2: 156 have finite 299-dimensional geo vectors and
`tok` is explicitly masked. Using this full-size MMS-1B configuration, synthetic
variable-length audio, FP16 AMP and four-step gradient accumulation, both mixed
`eng`/`tok` and all-`tok` batches completed finite backward passes and optimizer
updates. Initial AMP overflows were skipped while the dynamic loss scale
decreased, as in the standard trainer. This checks training startup, not a new
full training run.

```bash
# VoxLingua107, shared trainable conditioning (conf/train.yaml)
./run.sh --stage 4 --stop_stage 7

# Variant: pass the self-contained configuration directly
./run.sh --lid_config conf/tuning/independent_frozen.yaml --stage 4 --stop_stage 7

# Combined training
./run_combined.sh --stage 4 --stop_stage 7
```

Every YAML in `conf/tuning/` is a complete configuration that can be passed
directly with `--lid_config`; no merge step is needed.
The training schedules are 15,000 optimizer updates for
VoxLingua107 and 16,500 for combined training. Combined training balances datasets;
validation uses the language-based `catpow` sampler and does not require
`dataset2utt` or `utt2dataset` files. RIR augmentation is disabled and `rir_scp` is
null by default.

For recipe evaluation, download a converted model and point `--lid_exp` at it
(after preparing the corresponding waveform dumps):

```bash
python - <<'PY'
from pathlib import Path
from shutil import copyfile
from espnet_model_zoo.downloader import ModelDownloader

files = ModelDownloader().download_and_unpack("espnet/geolid_vl107only_shared_trainable")
destination = Path("exp/pretrained")
destination.mkdir(parents=True, exist_ok=True)
copyfile(files["train_config"], destination / "config.yaml")
copyfile(files["model_file"], destination / "model.pth")
PY
./run.sh --stage 6 --stop_stage 7 --skip_train true \
    --lid_exp exp/pretrained --inference_model model.pth
```

`ModelDownloader` resolves the bundled language-label path in the configuration.

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

The converted column reports measured utterance-level accuracy on all **1,609
VoxLingua107 dev utterances**, with correct counts in parentheses. All other
columns, including Macro Avg., are historical paper results and do not describe
new measurements of the converted checkpoints.

| Config | 🤗 HF Repo | VoxLingua107 dev (paper) | VoxLingua107 dev (converted) | Babel | FLEURS | ML-SUPERB2.0 Dev | ML-SUPERB2.0 Dialect | VoxPopuli | Macro Avg. |
| ------ | ---------- | ---------------------- | -------------------------- | ----- | ------ | --------------- | ------------------ | --------- | ---------- |
| `conf/tuning/independent_frozen.yaml` | [espnet/geolid_vl107only_independent_frozen](https://huggingface.co/espnet/geolid_vl107only_independent_frozen) | 94.2 | **94.28** (1517/1609) | 87.1 | 95.0 | 89.0 | 77.2 | 90.4 | 88.8 |
| `conf/tuning/independent_trainable.yaml` | [espnet/geolid_vl107only_independent_trainable](https://huggingface.co/espnet/geolid_vl107only_independent_trainable) | 93.7 | **93.78** (1509/1609) | 85.3 | 93.7 | 88.3 | 70.3 | 86.5 | 86.3 |
| `conf/tuning/shared_frozen.yaml` | [espnet/geolid_vl107only_shared_frozen](https://huggingface.co/espnet/geolid_vl107only_shared_frozen) | 94.3 | **94.34** (1518/1609) | 85.9 | 94.3 | 88.8 | 80.7 | 89.2 | 88.8 |
| `conf/train.yaml` | [espnet/geolid_vl107only_shared_trainable](https://huggingface.co/espnet/geolid_vl107only_shared_trainable) | 94.9 | **94.78** (1525/1609) | 87.7 | 93.5 | 89.3 | 78.8 | 89.5 | 88.9 |
| `conf/tuning/combined.yaml` | [espnet/geolid_combined_shared_trainable](https://huggingface.co/espnet/geolid_combined_shared_trainable) | 94.4 | **93.35** (1502/1609) | 95.4 | 97.7 | 88.6 | 86.8 | 99.0 | 93.7 |

These converted-model measurements used the standard `espnet2.bin.lid_inference`
CLI at revision `af67930739efa0bc7164beac8c79e556bd898a00`, GPU float32, batch
size 4, 8 data-loading workers, and full-length waveforms (`--use_preprocessor
false --fix_duration false`). Inputs were sorted by utterance ID, matching the
recipe's `wav.scp`. The environment used PyTorch 2.11.0+cu128, Transformers
5.14.1 and espnet-s3prl 2026.5.14. Batch composition matters because pooling and
normalization differ from the legacy implementation.

[Evaluation metadata and checkpoint hashes](results/voxlingua107_dev.json) record
the exact counts and settings. Each HF repository also provides `evaluation.json`.
In particular, the combined converted model scores 93.35%, below the historical
94.4%; full reproduction of the paper metrics is not established.

## Citation

```BibTex
@inproceedings{wang2025geolid,
  author={Qingzheng Wang and Hye-jin Shim and Jiancheng Sun and Shinji Watanabe},
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
