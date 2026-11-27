# URGENT24 audio quality assessment

Train Uni-VERSA to predict precomputed evaluation metrics for URGENT24 audio.
This recipe consumes externally prepared audio and scores. It does not download
the corpus or compute the target metrics. See the [AQA task guide](../../TEMPLATE/aqa1/README.md)
for the label format and shared stages.

## Published results

The [Uni-VERSA paper (Interspeech 2025)](https://www.isca-archive.org/interspeech_2025/shi25f_interspeech.pdf)
reports the following URGENT24 test results (Tables 2 and 3). Each entry is
utterance-level **LCC / SRCC** between predicted and target metric values;
higher is better. The average covers all 11 evaluation metrics in the paper.

| Features | Reference inputs | PESQ | MOS | WER | Average |
| --- | --- | --- | --- | --- | --- |
| Fbank | Audio + text | 0.78 / 0.80 | 0.72 / 0.72 | 0.57 / 0.53 | 0.76 / 0.73 |
| WavLM Large | Audio + text | 0.82 / 0.84 | 0.77 / 0.77 | 0.79 / 0.79 | 0.84 / 0.82 |
| WavLM Large | Audio only | 0.82 / 0.83 | 0.78 / 0.78 | 0.78 / 0.76 | 0.85 / 0.82 |
| WavLM Large | Text only | 0.84 / 0.86 | 0.79 / 0.79 | 0.78 / 0.76 | 0.85 / 0.83 |
| WavLM Large | None | 0.85 / 0.86 | 0.77 / 0.78 | 0.79 / 0.78 | 0.85 / 0.82 |

The paper partitions source utterances into train/dev/test sets in an 85:5:10
ratio, keeping outputs from the same source in the same split, and uses 16 kHz
audio. These are the original publication's results, not new measurements from
this revived recipe. In particular, the published setup freezes the SSL feature
extractor and trains for 50 epochs; the example configurations here use different
epoch limits and do not freeze WavLM by default. Matching the paper also requires
its data partition and metric labels. Synthetic smoke-test scores are not
benchmark results.

## Pretrained models

The [official Uni-VERSA collection](https://huggingface.co/collections/espnet/universa-6834e7c0a28225bffb6e2526)
contains models released by Jiatong Shi (`ftshijt`). The following releases are
relevant to the systems above. Use each release's saved architecture/configuration
when loading its weights; the local example configurations are for new training.
The release names say `wavlm_base`, but their saved frontend selects `wavlm_large`.

| Model | Reference inputs | Saved config and checkpoint directory | Weight file |
| --- | --- | --- | --- |
| [Fbank](https://huggingface.co/espnet/universa-base_urgent24_multi-metric/tree/7cd604c67a5a3e0a2cc79453c6cead9f17d99e9a) | Audio + text | `exp/universa_train_universa_raw_fs16000/` | `96epoch.pth` |
| [WavLM Large, full reference](https://huggingface.co/espnet/universa-wavlm_base_urgent24_multi-metric_fullref/tree/23c228a8f7cd658de16850b897b8c662fe815efa) | Audio + text | `update_exp/universa_train_universa_wavlm_freeze_raw_fs16000/` | `7epoch.pth` |
| [WavLM Large, audio reference](https://huggingface.co/espnet/universa-wavlm_base_urgent24_multi-metric_audioref/tree/fb24f25a303546a0857b39328d88141029c6f022) | Audio only | `update_exp/universa_train_universa_wavlm_ref_audio_raw_fs16000/` | `15epoch.pth` |
| [WavLM Large, text reference](https://huggingface.co/espnet/universa-wavlm_base_urgent24_multi-metric_textref/tree/d84d3a141989f9e65f3121a33b1c64b9cd76db8b) | Text only | `update_exp/universa_train_universa_wavlm_text_raw_fs16000/` | `17epoch.pth` |
| [WavLM Large, no reference](https://huggingface.co/espnet/universa-wavlm_base_urgent24_multi-metric_noref/tree/6abec38c0971c2c0feebbe39ee2f94c9c21357c4) | None | `update_exp/universa_train_universa_wavlm_noref_raw_fs16000/` | `13epoch.pth` |

Each directory in the table contains `config.yaml`. Links pin the inspected
Hugging Face revisions. The published results above describe the paper's systems;
these checkpoint files have not been re-benchmarked with this recipe.

All four suffixed WavLM packages include the metric vocabulary at
`dump_ark/raw/train_update/metric2id` and a BPE model at
`data/token_list/bpe_unigram500/bpe.model`. Resolve these saved paths relative to
the downloaded package before loading. Their saved configurations freeze
`frontend.upstream` and set `max_epoch: 100`; batch size is 16 for the reference
variants and 32 for no-reference. Their `meta.yaml` records ESPnet `202412`,
Python `3.9.18`, and PyTorch `2.6.0.dev20241210+cu124` as the release environment.
These historical versions are provenance, not installation requirements for
current ESPnet.

The released vocabulary contains 13 targets in this order:

```text
dns_p808 f0corr f0rmse mcd mos pesq sheet_ssqa si_snr speech_bert spk_similarity stoi utmos wer
```

The paper reports 11 evaluation metrics, so its average excludes the additional
F0-RMSE and MCD outputs. Preserve the package's full vocabulary and ordering when
loading its checkpoint. These four WavLM packages provide training curves but no
machine-readable held-out evaluation results; the correlation table above is
sourced from the paper, not inferred from the weights or training curves.

The older Fbank package in the table and the
[unsuffixed WavLM package](https://huggingface.co/espnet/universa-wavlm_base_urgent24_multi-metric/tree/16184f434b0da62a2a723ee6afd0b74b8439fb0b)
omit their referenced metric vocabularies and BPE models. Prefer the complete
`_fullref` WavLM release above. Using either older package requires restoring its
exact original assets; do not assume a newer release has the same metric ordering
or tokenizer.

For inference with a complete package, use `espnet2.bin.aqa_inference` with
`--train_config` pointing to the resolved saved configuration and `--model_file`
pointing to the corresponding weight file. Retain the saved reference settings
and install ESPnet's S3PRL fork for WavLM, as described below. For fine-tuning,
use the original resolved configuration with `--init_param` and preserve its
`freeze_param`; an epoch weight file is not a resumable trainer checkpoint.
See the [AQA task guide](../../TEMPLATE/aqa1/README.md#arecho-and-legacy-models)
for legacy configuration compatibility.

## Data preparation and training

Prepare `data/train`, `data/dev`, and `data/test`. Each split must contain
`wav.scp`, `utt2spk`, and `metric.scp`, keyed by the same utterance IDs.
The default configuration also requires `ref_wav.scp` and `text`. Omit them
when running with `--use_ref_wav false` and `--use_ref_text false`, respectively.
Audio entries use Kaldi SCP format; `ref_wav.scp` may use
`utt_id None` for a missing reference. Text entries contain the reference
transcript. Each `metric.scp` entry contains an utterance ID and a JSON object
of numeric metric values, for example:

```text
utt1 {"pesq": 2.1, "wer": 10.0}
```

Omit unavailable metric labels from the JSON object. `local/data.sh` checks
that the files exist, sorts them, and generates `spk2utt` from `utt2spk`.
Stage 3 discovers metric names from the training labels.

From this directory, run the baseline with log-mel features:

```bash
./run.sh
```

To use WavLM Large features, install S3PRL with the ESPnet tools installer
(`cd ../../../tools && make s3prl.done`), then run:

```bash
./run.sh --train_config conf/tuning/train_wavlm.yaml
```

The WavLM frontend downloads pretrained weights on first use. Both configurations
use 16 kHz audio and enable reference text/audio by default. The default is one training GPU;
use `--ngpu 0` for CPU training. Override shared driver options through `run.sh`,
for example `--stage 6 --stop_stage 7` to collect statistics and train after data
preparation. These are corpus training configurations, not CI debug settings.
