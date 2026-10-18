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
is not emulated. The results below are historical paper results, not a new corpus
evaluation of these converted models. The models inherit **CC-BY-NC-4.0** from
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
dumps. Missing lang2vec languages are reported as an error; zero vectors are
never silently substituted.

```bash
# VoxLingua107, shared trainable conditioning (conf/train.yaml)
./run.sh --stage 4 --stop_stage 7

# Variant: merge only these changed settings onto conf/train.yaml
./run.sh --tuning_config conf/tuning/independent_frozen.yaml --stage 4 --stop_stage 7

# Combined training
./run_combined.sh --stage 4 --stop_stage 7
```

`conf/tuning/` contains only overrides. `run.sh` writes the merged configuration
to `exp/configs/`. The training schedules are 15,000 optimizer updates for
VoxLingua107 and 16,500 for combined training. Combined training balances datasets;
validation uses the language-based `catpow` sampler and does not require
`dataset2utt` or `utt2dataset` files. RIR augmentation is disabled and `rir_scp` is
null by default.

For recipe evaluation, download a converted model and point `--lid_exp` at it
(after preparing the corresponding waveform dumps):

```bash
hf download espnet/geolid_vl107only_shared_trainable --local-dir exp/pretrained
./run.sh --stage 6 --stop_stage 7 --skip_train true \
    --lid_exp exp/pretrained --inference_model model.pth
```

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
| `conf/tuning/independent_frozen.yaml` | [espnet/geolid_vl107only_independent_frozen](https://huggingface.co/espnet/geolid_vl107only_independent_frozen) | 94.2         | 87.1  | 95.0   | 89.0             | 77.2                 | 90.4      | 88.8       |
| `conf/tuning/independent_trainable.yaml` | [espnet/geolid_vl107only_independent_trainable](https://huggingface.co/espnet/geolid_vl107only_independent_trainable) | 93.7         | 85.3  | 93.7   | 88.3             | 70.3                 | 86.5      | 86.3       |
| `conf/tuning/shared_frozen.yaml` | [espnet/geolid_vl107only_shared_frozen](https://huggingface.co/espnet/geolid_vl107only_shared_frozen) | 94.3         | 85.9  | 94.3   | 88.8             | 80.7                 | 89.2      | 88.8       |
| `conf/train.yaml` | [espnet/geolid_vl107only_shared_trainable](https://huggingface.co/espnet/geolid_vl107only_shared_trainable) | 94.9         | 87.7  | 93.5   | 89.3             | 78.8                 | 89.5      | 88.9       |
| `conf/tuning/combined.yaml` | [espnet/geolid_combined_shared_trainable](https://huggingface.co/espnet/geolid_combined_shared_trainable) | 94.4         | 95.4  | 97.7   | 88.6             | 86.8                 | 99.0      | 93.7       |





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
