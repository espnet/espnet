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
through the standard installers. The recipe checks these dependencies in
`local/path.sh` and reports installation commands when any are missing:

```bash
cd tools
make s3prl.done transformers.done lang2vec.done
cd ../egs2/geolid/lid1
python3 -m pip install 'datasets>=2.16,<4' pycountry
```

S3PRL's `hf_wav2vec2_custom` upstream imports Transformers; `espnet-s3prl` does
not install it as a dependency. No GeoLID library code imports Transformers or
lang2vec directly. `lang2vec` is needed only to prepare training target tables.
The recipe-specific `datasets<4` and `pycountry` requirements are for corpus
preparation; the shared FLEURS loader requires dataset scripts removed in
`datasets` 4. Babel preparation, which is also required for combined training,
additionally needs `sph2pipe`; see the installation command in the
[data preparation guide](local/README.md#dependencies-and-corpus-paths).

The recipe uses the stock `S3prlFrontend`, configured in `conf/train.yaml`:

```yaml
frontend: s3prl
frontend_conf:
  frontend_conf:
    upstream: hf_wav2vec2_custom
    path_or_url: facebook/mms-1b
  multilayer_feature: true
```

`ESPnetLIDModel` optionally replaces encoder layers 32, 36, 40 and 44
(zero-based) with `Lang2VecConditionedLayer` wrappers. Each wrapper runs the
original SSL layer, pools its output, predicts a lang2vec vector and adds the
projected vector to the frame representations before the next SSL layer. The
model creates the projections, optionally shares them across layers, and collects
the intermediate predictions for auxiliary losses. `config.layerdrop` is set to
zero so these layers always execute. Attention and waveform preprocessing remain
in stock Transformers and S3PRL. No custom Transformers class or
checkpoint-local model code is needed.

The configurations use the standard `model: espnet`; lang2vec supervision and
conditioning are optional features configured through `model_conf`. Classification
uses the existing `loss: aamsoftmax_sc_topk` without modifying its three-value return contract.
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
from `loss_conf` to `model_conf`. It also updates earlier `model: lang2vec`
configurations to `model: espnet`, preserving their auxiliary loss settings
and already converted parameter names. Legacy key handling exists only in
this conversion script.

### Training and recipe evaluation

Use `run.sh` to select either setup. The default prepares VoxLingua107 and
writes `data/voxlingua107_lang2vec.txt`. Before combined preparation, arrange the
licensed Babel audio and downloaded VoxPopuli audio as described in the
[data preparation guide](local/README.md).

```bash
# Default: VoxLingua107, shared trainable conditioning
./run.sh

# Combined: prepare all five corpora, filter Babel, combine, train and evaluate
./run.sh --setup combined

# Optional corpus locations (defaults: downloads/babel and downloads/voxpopuli)
./run.sh --setup combined \
    --local_data_opts "--babel_path /corpora/babel --voxpopuli_path /corpora/voxpopuli"

# Prepare data only; stage 3 also filters Babel and creates the combined dump
./run.sh --setup combined --stop_stage 3

# Resume from existing combined dumps without repeating preparation
./run.sh --setup combined --stage 4 --stop_stage 7

# Variant: pass the self-contained configuration directly
./run.sh --lid_config conf/tuning/independent_frozen.yaml --stage 4 --stop_stage 7
```

Both setups use the shared `lid.sh` stages. For combined preparation,
`local/prepare_combined.sh` runs stages 1–3 separately for each corpus, then keeps
Babel utterances of at least 10 seconds and combines the training dumps before
stage 4. Stopping at stage 1 or 2 does not filter or combine incomplete dumps;
resuming at stage 3 uses the prepared data directories.

Combined training defaults to `conf/tuning/combined.yaml`,
`dump/raw/train_all_no_filter_lang`, validation on `dev_ml_superb2_lang`, and
experiment directory `exp_combined`. It evaluates VoxLingua107 dev, VoxPopuli
test, FLEURS test, ML-SUPERB2 dev and dialect dev, and filtered Babel dev.
The VoxLingua107-only setup uses `conf/train.yaml` and experiment directory
`exp_voxlingua107`. Both setups use raw 16 kHz audio without speed perturbation.

`run.sh` declares its options and uses the standard `utils/parse_options.sh`.
It accepts `--setup`, `--stage`, `--stop_stage`, `--ngpu`, `--nj`, `--python`,
`--local_data_opts` and `--lid_config`. Set other pipeline options by editing the
explicit `lid.sh` arguments in `run.sh`; corpus preparation options are in
`local/prepare_combined.sh`. The preparation helper retains each corpus's own
split names and writes its dumps under `dump/raw`.

`local/combine.sh` creates the dataset balancing maps and
`data/combined_lang2vec.txt` using `local/prepare_lang2vec.py --exclude tok`.
URIEL has no geo vector for Toki Pona: all 157 classification labels are
retained, but the table has only the 156 observed targets. Missing features for
any other language cause target preparation to fail. Each target-table row is
`<lang> <v1> ... <vN>`. The generic LID preprocessor reads the table via
`lang2vec_table`, without querying lang2vec, and masks missing targets out of
both auxiliary losses. All-masked batches use classification loss alone.

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

For recipe evaluation, download a converted model and call the shared `lid.sh`
directly with `--lid_exp` pointing at it (after preparing the corresponding
waveform dumps):

```bash
python - <<'PY'
from pathlib import Path
from shutil import copyfile
from espnet_model_zoo.downloader import ModelDownloader

files = ModelDownloader().download_and_unpack("espnet/geolid_vl107only_shared_trainable")
destination = Path("exp_voxlingua107/pretrained")
destination.mkdir(parents=True, exist_ok=True)
copyfile(files["train_config"], destination / "config.yaml")
copyfile(files["model_file"], destination / "model.pth")
PY
./lid.sh --stage 6 --stop_stage 7 --skip_train true \
    --feats_type raw --fs 16k --expdir exp_voxlingua107 \
    --train_set train_voxlingua107_lang --valid_set dev_voxlingua107_lang \
    --test_sets dev_voxlingua107_lang --inference_batch_size 1 --extract_embd true \
    --lid_exp exp_voxlingua107/pretrained --inference_model model.pth
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

The legacy-reference and converted columns report measured utterance-level
accuracy on all **1,609 VoxLingua107 dev utterances**, with correct counts in
parentheses for the converted models. The legacy reference evaluates the original
published weights through the earlier HF implementation, preserving legacy
normalization and pooling, with **batch size 4**. The converted models use the
current ESPnet implementation with **batch size 1**. Both evaluations use the same
utterance IDs and reference labels; the comparison includes both runtime and
batching differences. The other dataset columns in the paper rows, including
Macro Avg., are historical results. **Every combined paper number belongs to
the earlier epoch-24 checkpoint; the released combined checkpoint is epoch 31.**
The historical combined row is separated from the released model below.

| Config | 🤗 HF Repo | VoxLingua107 dev (paper) | VoxLingua107 dev (legacy reference) | VoxLingua107 dev (converted) | Babel | FLEURS | ML-SUPERB2.0 Dev | ML-SUPERB2.0 Dialect | VoxPopuli | Macro Avg. |
| ------ | ---------- | ---------------------- | --------------------------------- | -------------------------- | ----- | ------ | --------------- | ------------------ | --------- | ---------- |
| `conf/tuning/independent_frozen.yaml` | [espnet/geolid_vl107only_independent_frozen](https://huggingface.co/espnet/geolid_vl107only_independent_frozen) | 94.2 | 94.22 | **94.28** (1517/1609) | 87.1 | 95.0 | 89.0 | 77.2 | 90.4 | 88.8 |
| `conf/tuning/independent_trainable.yaml` | [espnet/geolid_vl107only_independent_trainable](https://huggingface.co/espnet/geolid_vl107only_independent_trainable) | 93.7 | 93.72 | **93.78** (1509/1609) | 85.3 | 93.7 | 88.3 | 70.3 | 86.5 | 86.3 |
| `conf/tuning/shared_frozen.yaml` | [espnet/geolid_vl107only_shared_frozen](https://huggingface.co/espnet/geolid_vl107only_shared_frozen) | 94.3 | 94.34 | **94.28** (1517/1609) | 85.9 | 94.3 | 88.8 | 80.7 | 89.2 | 88.8 |
| `conf/train.yaml` | [espnet/geolid_vl107only_shared_trainable](https://huggingface.co/espnet/geolid_vl107only_shared_trainable) | 94.9 | 94.90 | **94.84** (1526/1609) | 87.7 | 93.5 | 89.3 | 78.8 | 89.5 | 88.9 |
| Combined, paper (epoch 24; earlier weights) | [Archived evaluation reports][combined-original-evaluations] | 94.4 | — | — | 95.4 | 97.7 | 88.6 | 86.8 | 99.0 | 93.7 |
| `conf/tuning/combined.yaml` (released epoch 31) | [espnet/geolid_combined_shared_trainable](https://huggingface.co/espnet/geolid_combined_shared_trainable) | — | 93.35 | **93.29** (1501/1609) | — | 97.51 | 88.69 | 85.26 | 98.98 | — |

Macro Avg. is the unweighted mean of the six test sets' utterance-level
accuracies, rather than a mean over languages or accuracy pooled over all
utterances. New measurements use the full-precision accuracies before rounding;
historical rows retain the values reported in the paper. A dash denotes an
unreported measurement; no new six-set Macro Avg. is reported until all six
measurements exist.

The released combined row reports new measurements at ESPnet revision
`07dc3256fbe6ab542a260314f9e3d9a0bdc9de9a`. The [combined evaluation
report](https://huggingface.co/espnet/geolid_combined_shared_trainable/blob/0461e9528b33bea08a29979cce60e685f04384c4/evaluations/20261027/report.md)
includes the reproduction command, source revisions, complete predictions,
reference labels and inference logs. The run covers all 1,609 VoxLingua107 dev,
77,810 FLEURS test, 23,609 ML-SUPERB 2.0 dev, 7,095 dialect-dev and 16,991
VoxPopuli test utterances. VoxPopuli uses the official test partition; the
recipe now reads `asr_test.tsv` instead of reusing development data.
ML-SUPERB uses the shared recipe's language normalization and excludes its
283 Norwegian dev rows. These split sizes, archived error IDs and reference
labels reproduce the historical reports' per-language accuracies. The old
reports do not establish byte-for-byte identity of all historical audio.

FLEURS uses the standard CLI's four-GPU multiprocessing mode, with one
utterance per forward pass and 8 workers per process. All 4,000 predictions
from the initial single-GPU run match the parallel run exactly. The other
test sets use one GPU.

**Babel remains unevaluated:** its licensed dev audio and original complete
manifest are unavailable in this evaluation environment. The released-model
Babel cell and six-set Macro Avg. remain blank pending that evaluation.

The new combined VoxLingua107 run reproduces all 1,609 previous predictions
(1,501 correct). The four VoxLingua107-only models' converted results retain
their original evaluation revision `301372e4da311d2a3d9574a5a43b3037ae6e67c1`.

These converted-model measurements use the standard `espnet2.bin.lid_inference`
CLI, GPU float32, **batch size 1 per forward pass**, 8 data-loading workers
per inference process and full-length waveforms
(`--use_preprocessor false --fix_duration false`). Inputs are sorted
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

Each HF repository preserves its earlier VoxLingua107 counts, runtime settings,
implementation revision and checkpoint hash in `evaluation.json`. The new
combined-model measurements and per-set evidence are under `evaluations/20261027/`.

### Origin of the combined model's paper results

**All six combined-model paper accuracies and their Macro Avg. were obtained
with the earlier epoch-24 checkpoint (48,000 training batches), not the released
epoch-31 checkpoint (62,000 batches).** This applies to VoxLingua107 dev, Babel,
FLEURS, ML-SUPERB 2.0 dev, ML-SUPERB 2.0 dialect and VoxPopuli. The four
VoxLingua107-only models' legacy references remain consistent with their paper
VoxLingua107 dev results.

The [archived combined evaluation reports][combined-original-evaluations] give
94.41% on VoxLingua107 dev, 95.38% on Babel, 97.73% on FLEURS, 88.61% on
ML-SUPERB 2.0 dev, 86.82% on its dialect set and 98.95% on VoxPopuli. These round
to the combined paper row, whose reported Macro Avg. is 93.7%. All six inference
runs started between **00:34 and 02:37 CDT on June 2, 2025**; the last run, Babel,
finished at **02:55 CDT**. The logs record full-length audio, GPU float32,
batch size 4 and 32 workers, loading `valid.accuracy.best.pth`.

The [training log][combined-training-first] shows that the most recent best
checkpoint then was **epoch 24 / 48,000 training batches**. The
[resumed training log][combined-training-resumed] records a new best at
**epoch 31 / 62,000 batches**, at **11:22 CDT** that day, after the archived
evaluations. The published checkpoint's 41 BatchNorm `num_batches_tracked`
counters are all 62,000, consistent with that later checkpoint. Although
Section IV-C of the paper specifies a 62k checkpoint, every archived combined
score predates it. The epoch-24 weights were evaluated, while the later epoch-31
weights were released under the same filename.

This is a discrepancy between the historical results and the released checkpoint.
On VoxLingua107 dev, the gap from 94.4% is already present in the
**93.35% legacy reference**, before
conversion to the current **93.29%** model. The historical inference log did not
record a checkpoint hash, and the earlier checkpoint has not been recovered, so
its exact weight identity cannot be verified by checksum or a new run. The
epoch assignment above follows the archived training and inference logs.
The historical audio files were not checksummed either. All six combined paper
numbers and Macro Avg. remain historical epoch-24 results, not measurements of
the released epoch-31 weights.

[combined-original-evaluations]: https://huggingface.co/espnet/geolid_combined_shared_trainable/tree/7dc4bf988efff29f725bf34b1caa821c27359793/exp_combined/lid_mms_ecapa_upcon_32_44_it0.4_shared_trainable_raw/inference/valid.accuracy.best
[combined-training-first]: https://huggingface.co/espnet/geolid_combined_shared_trainable/blob/7dc4bf988efff29f725bf34b1caa821c27359793/exp_combined/lid_mms_ecapa_upcon_32_44_it0.4_shared_trainable_raw/train.4.log
[combined-training-resumed]: https://huggingface.co/espnet/geolid_combined_shared_trainable/blob/7dc4bf988efff29f725bf34b1caa821c27359793/exp_combined/lid_mms_ecapa_upcon_32_44_it0.4_shared_trainable_raw/train.3.log

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
