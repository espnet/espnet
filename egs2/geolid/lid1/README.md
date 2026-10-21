# README

[Paper](https://arxiv.org/pdf/2508.17148)

This recipe implements GeoLID with pretrained
[MMS-1B](https://huggingface.co/facebook/mms-1b) features and an ECAPA-TDNN
embedding extractor. Language-level geographic vectors from lang2vec supervise
auxiliary prediction heads during training. Their predictions condition
intermediate SSL representations; inference consumes speech only.

See *Geolocation-Aware Robust Spoken Language Identification*
([paper](https://arxiv.org/pdf/2508.17148)) for the method and historical results.

## Usage

After setting up ESPnet's tools environment, install the optional dependencies
through the standard installers:

```bash
cd tools
make s3prl.done transformers.done lang2vec.done
cd ../egs2/geolid/lid1
python -m pip install 'datasets>=2.16,<4' pycountry
```

S3PRL's `hf_wav2vec2_custom` upstream imports Transformers; `espnet-s3prl` does
not install it as a dependency. No GeoLID library code imports Transformers or
lang2vec directly. `lang2vec` is needed only to prepare training target tables.
The recipe-specific `datasets<4` and `pycountry` requirements are for corpus
preparation; the shared FLEURS loader requires dataset scripts removed in
`datasets` 4.

The recipe uses the stock `S3prlFrontend`, configured in `conf/train.yaml`:

```yaml
frontend: s3prl
frontend_conf:
  frontend_conf:
    upstream: hf_wav2vec2_custom
    path_or_url: facebook/mms-1b
  multilayer_feature: true
```

`ESPnetLang2VecLIDModel` replaces encoder layers 32, 36, 40 and 44
(zero-based) with `Lang2VecConditionedLayer` wrappers. Each wrapper runs the
original SSL layer, pools its output, predicts a lang2vec vector and adds the
projected vector to the frame representations before the next SSL layer. The
model creates the projections, optionally shares them across layers, and collects
the intermediate predictions for auxiliary losses. `config.layerdrop` is set to
zero so these layers always execute. Attention and waveform preprocessing remain
in stock Transformers and S3PRL. No custom Transformers class or
checkpoint-local model code is needed.

The model is selected with `model: lang2vec`. Classification uses the existing
`loss: aamsoftmax_sc_topk` without modifying its three-value return contract.
A linear head in the model predicts language features from the final utterance
embedding. The intermediate heads predict the same targets. `lang2vec_loss:
mse` uses mean squared error for continuous geo vectors; `bce` uses binary
cross-entropy with logits for binary `*_knn` targets. The auxiliary loss mixes
intermediate and downstream losses with `inter_lang2vec_loss_weight`, then mixes
that result with classification using `lang2vec_weight`.

Language features are training targets. At inference, conditioning uses features
predicted from speech; no language or geographic metadata is supplied.

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
Each repo includes conversion provenance in `conversion.json` and strict-loading
and full-dev inference validation in `validation.json`.

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
projection names, and the earlier converted layout. It moves the auxiliary head
from `loss.lang2vec_head.0.*` to `lang2vec_head.*` and moves auxiliary options
from `loss_conf` to `model_conf`. Legacy key handling exists only in this
conversion script.

### Training and recipe evaluation

Follow [local/README.md](local/README.md) for the corpus preparation commands.
There is one entrypoint, `run.sh`. Stage 1 prepares VoxLingua107 by default and
writes `data/voxlingua107_lang2vec.txt`. A target table contains one language
per row: `<lang> <v1> ... <vN>`. The generic LID preprocessor reads the table via
`lang2vec_table` and emits targets and a mask; it does not query lang2vec.

```bash
# Default: VoxLingua107, shared trainable conditioning
./run.sh

# Variant: pass the self-contained configuration directly
./run.sh --lid_config conf/tuning/independent_frozen.yaml --stage 4 --stop_stage 7
```

For combined training, prepare each corpus and run stage 3 for its train/dev/test
splits first. For example:

```bash
./run.sh --stage 3 --stop_stage 3 \
    --train_set train_fleurs_lang --valid_set dev_fleurs_lang \
    --test_sets test_fleurs_lang
# Repeat for the other corpora as listed in local/README.md.
bash local/filter_babel_over_10s.sh
bash local/combine.sh
./run.sh --lid_config conf/tuning/combined.yaml \
    --train_set train_all_no_filter_lang --valid_set dev_ml_superb2_lang \
    --test_sets "dev_voxlingua107_lang test_voxpopuli_lang test_fleurs_lang dev_ml_superb2_lang dev_dialect_ml_superb2_lang dev_babel_over_10s_lang" \
    --expdir exp_combined --tsne_set train_all_no_filter_lang \
    --stage 4 --stop_stage 7
```

`local/combine.sh` must run between stages 3 and 4. It combines the waveform
dumps and prepares `data/combined_lang2vec.txt` using `local/prepare_lang2vec.py
--exclude tok`. URIEL has no geo vector for Toki Pona: all 157 classification
labels are retained, but the table has only the 156 observed targets. Missing
features for any other language cause target preparation to fail. The
preprocessor masks languages absent from the table out of both auxiliary
losses; all-masked batches use classification loss alone.

To regenerate target tables from existing dumps:

```bash
python local/prepare_lang2vec.py \
    --lang2utt dump/raw/train_voxlingua107_lang/lang2utt \
    --output data/voxlingua107_lang2vec.txt
python local/prepare_lang2vec.py \
    --lang2utt dump/raw/train_all_no_filter_lang/lang2utt \
    --output data/combined_lang2vec.txt --exclude tok
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

Here, a **converted model** is an original pretrained GeoLID checkpoint adapted
to the current ESPnet implementation by remapping parameter keys and removing
verified duplicate aliases. The learned tensor values are unchanged, and no
retraining was performed. These models use current ESPnet normalization and
pooling, so their predictions can differ from those of the original implementation.

The converted column reports measured utterance-level accuracy on all **1,609
VoxLingua107 dev utterances**, with correct counts in parentheses. All other
columns, including Macro Avg., are historical paper results and do not describe
new measurements of the converted checkpoints.

| Config | 🤗 HF Repo | VoxLingua107 dev (paper) | VoxLingua107 dev (converted) | Babel | FLEURS | ML-SUPERB2.0 Dev | ML-SUPERB2.0 Dialect | VoxPopuli | Macro Avg. |
| ------ | ---------- | ---------------------- | -------------------------- | ----- | ------ | --------------- | ------------------ | --------- | ---------- |
| `conf/tuning/independent_frozen.yaml` | [espnet/geolid_vl107only_independent_frozen](https://huggingface.co/espnet/geolid_vl107only_independent_frozen) | 94.2 | **94.28** (1517/1609) | 87.1 | 95.0 | 89.0 | 77.2 | 90.4 | 88.8 |
| `conf/tuning/independent_trainable.yaml` | [espnet/geolid_vl107only_independent_trainable](https://huggingface.co/espnet/geolid_vl107only_independent_trainable) | 93.7 | **93.78** (1509/1609) | 85.3 | 93.7 | 88.3 | 70.3 | 86.5 | 86.3 |
| `conf/tuning/shared_frozen.yaml` | [espnet/geolid_vl107only_shared_frozen](https://huggingface.co/espnet/geolid_vl107only_shared_frozen) | 94.3 | **94.28** (1517/1609) | 85.9 | 94.3 | 88.8 | 80.7 | 89.2 | 88.8 |
| `conf/train.yaml` | [espnet/geolid_vl107only_shared_trainable](https://huggingface.co/espnet/geolid_vl107only_shared_trainable) | 94.9 | **94.84** (1526/1609) | 87.7 | 93.5 | 89.3 | 78.8 | 89.5 | 88.9 |
| `conf/tuning/combined.yaml` | [espnet/geolid_combined_shared_trainable](https://huggingface.co/espnet/geolid_combined_shared_trainable) | 94.4 | **93.29** (1501/1609) | 95.4 | 97.7 | 88.6 | 86.8 | 99.0 | 93.7 |

These converted-model measurements use the standard `espnet2.bin.lid_inference`
CLI, GPU float32, **batch size 1**, 8 data-loading workers and full-length
waveforms (`--use_preprocessor false --fix_duration false`). Inputs are sorted
by utterance ID, matching the recipe's `wav.scp`. The environment uses PyTorch
2.11.0+cu128, Transformers 5.14.1 and espnet-s3prl 2026.5.14. The recipe defaults
to `--inference_batch_size 1` for reproducible per-utterance evaluation.

`utterance_mvn` and final `chn_attn_stat` pooling both mask padding. The remaining
batch dependence arises **before final pooling**:

- S3PRL reports `ceil(samples / 320)` frames and extends the full batch's hidden
  states by repeating its last frame. For a shorter item, frames beyond its
  valid Wav2Vec2 convolution output can instead be padded-query outputs from
  the longer batch. The utterance-level length mask includes these extra frames.
- ECAPA-TDNN does not take frame lengths: its temporal convolutions and SE
  global averages operate over the padded time axis. Adding zero padding to
  identical valid input features changes the valid encoded features and the
  embedding, even when final pooling receives the correct length mask.

These effects were isolated using the same speech in single-utterance and
mixed-length batches, and by padding fixed frontend features before ECAPA.
Masking only the final pooling cannot undo those upstream feature changes.

Each HF repository provides counts, runtime settings, implementation revision
and checkpoint hashes in `evaluation.json`. The combined converted model scores
**93.29%**, below the historical 94.4%; full reproduction of the paper metrics
is not established. Other dataset columns remain historical paper results.

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
