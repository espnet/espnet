# GeoLID

Recipe for [Geolocation-Aware Robust Spoken Language Identification](https://arxiv.org/pdf/2508.17148),
using MMS-1B features and ECAPA-TDNN with auxiliary lang2vec prediction and
conditioning. Inference consumes speech only.

## Installation

After setting up ESPnet's tools environment, run from the repository root:

```bash
make -C tools s3prl.done transformers.done lang2vec.done
python3 -m pip install 'datasets>=2.16,<4' pycountry
make -C tools sph2pipe  # Required for Babel and combined preparation
cd egs2/geolid/lid1
```

Transformers is required by S3PRL's `hf_wav2vec2_custom` upstream; lang2vec,
datasets and pycountry are used for data preparation. See the
[corpus paths and preparation steps](local/README.md).

## Usage

```bash
./run.sh  # VoxLingua107, shared trainable conditioning
./run.sh --setup combined  # Prepare all corpora, combine, train and evaluate
./run.sh --lid_config conf/tuning/independent_frozen.yaml

# Prepare combined data only, then resume training/evaluation
./run.sh --setup combined --stop_stage 3
./run.sh --setup combined --stage 4 --stop_stage 7
```

The experiment directories are `exp_voxlingua107` and `exp_combined`.
Combined stages 1–3 prepare each corpus, filter Babel to at least 10 seconds,
and combine the dumps automatically. All tuning configurations are self-contained.
`run.sh --help` lists supported options; edit its explicit `lid.sh` arguments
for other pipeline settings.

## Inference

The published models were converted without retraining and are licensed under
CC-BY-NC-4.0, inherited from [MMS-1B](https://huggingface.co/facebook/mms-1b).
After installing this recipe's ESPnet revision:

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

## Datasets

The VoxLingua107-only setup covers 107 training languages; combined training
uses all five corpora and 157 languages.

| Dataset | Domain      | #Langs. Train/Test | Dialect | Training Setup (VL107-only) | Training Setup (Combined) |
| ------------------------------------------------------------ | ----------- | ------------------ | ------- | --------------------------- | ------------------------- |
| [VoxLingua107](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/) | YouTube     | 107/33             | No      | Seen                        | Seen                      |
| [Babel](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=31a13cefb42647e924e0d2778d341decc44c40e9) | Telephone   | 25/25              | No      | Unseen                      | Seen                      |
| [FLEURS](https://huggingface.co/datasets/google/xtreme_s)    | Read speech | 102/102            | No      | Unseen                      | Seen                      |
| [ML-SUPERB 2.0](https://huggingface.co/datasets/espnet/ml_superb_hf) | Mixed       | 137/(137, 8)       | Yes     | Unseen                      | Seen                      |
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) | Parliament  | 16/16              | No      | Unseen                      | Seen                      |

## Results

Utterance-level accuracy (%). The legacy and converted columns measure
VoxLingua107 dev with batch sizes 4 and 1, respectively. Other columns in the
first four rows are historical paper results; the released combined row contains
new measurements. See [evaluation settings, evidence and limitations](EVALUATION.md).

| Config | 🤗 HF Repo | VoxLingua107 dev (paper) | VoxLingua107 dev (legacy reference) | VoxLingua107 dev (converted) | Babel | FLEURS | ML-SUPERB2.0 Dev | ML-SUPERB2.0 Dialect | VoxPopuli | Macro Avg. |
| ------ | ---------- | ---------------------- | --------------------------------- | -------------------------- | ----- | ------ | --------------- | ------------------ | --------- | ---------- |
| `conf/tuning/independent_frozen.yaml` | [espnet/geolid_vl107only_independent_frozen](https://huggingface.co/espnet/geolid_vl107only_independent_frozen) | 94.2 | 94.22 | **94.28** (1517/1609) | 87.1 | 95.0 | 89.0 | 77.2 | 90.4 | 88.8 |
| `conf/tuning/independent_trainable.yaml` | [espnet/geolid_vl107only_independent_trainable](https://huggingface.co/espnet/geolid_vl107only_independent_trainable) | 93.7 | 93.72 | **93.78** (1509/1609) | 85.3 | 93.7 | 88.3 | 70.3 | 86.5 | 86.3 |
| `conf/tuning/shared_frozen.yaml` | [espnet/geolid_vl107only_shared_frozen](https://huggingface.co/espnet/geolid_vl107only_shared_frozen) | 94.3 | 94.34 | **94.28** (1517/1609) | 85.9 | 94.3 | 88.8 | 80.7 | 89.2 | 88.8 |
| `conf/train.yaml` | [espnet/geolid_vl107only_shared_trainable](https://huggingface.co/espnet/geolid_vl107only_shared_trainable) | 94.9 | 94.90 | **94.84** (1526/1609) | 87.7 | 93.5 | 89.3 | 78.8 | 89.5 | 88.9 |
| Combined, paper (epoch 24; earlier weights) | [Archived evaluation reports](EVALUATION.md#combined-checkpoint-provenance) | 94.4 | — | — | 95.4 | 97.7 | 88.6 | 86.8 | 99.0 | 93.7 |
| `conf/tuning/combined.yaml` (released epoch 31) | [espnet/geolid_combined_shared_trainable](https://huggingface.co/espnet/geolid_combined_shared_trainable) | — | 93.35 | **93.29** (1501/1609) | — | 97.51 | 88.69 | 85.26 | 98.98 | — |

All six combined paper scores and Macro Avg. are from **epoch 24**; the released
weights are **epoch 31**. The earlier weights have not been recovered. Babel
audio and its complete evaluation manifest are unavailable here, so the released
Babel score and six-set Macro Avg. remain unreported. Macro Avg. gives each of
the six test sets equal weight.

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
