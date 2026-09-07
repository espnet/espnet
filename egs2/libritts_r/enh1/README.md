# ESPnet-Sidon: feature-predictor speech restoration

An ESPnet reproduction of **Sidon** ([arXiv:2509.17052](https://arxiv.org/abs/2509.17052)).
The model predicts the clean w2v-BERT 2.0 layer-8 hidden state from degraded
speech and vocodes it to 48 kHz with the official Sidon decoder.

## Extra dependencies

These are **not** installed by `tools/` and are not ESPnet dependencies:

| Package | Needed by | Install |
|---|---|---|
| `peft` | LoRA adapters on the w2v-BERT student (stages 4-5) | `pip install peft` |
| `pyroomacoustics` | RIR pool generation (stage 3) | `pip install pyroomacoustics` |
| `versa` | VERSA scoring (stage 8) | `tools/installers/install_versa.sh` |
| NISQA *(optional)* | `local/score.py --nisqa_model` | clone [NISQA](https://github.com/gabrielmittag/NISQA), add to `PYTHONPATH` |

`transformers` supplies the w2v-BERT 2.0 backbone, `WavLMForXVector` for
speaker similarity, and `facebook/mms-1b-all` for WER. All three are fetched
from the Hub on first use, so stages 4-7 need network access (or a pre-warmed
`HF_HOME`) on whichever node runs them.

## Data

Set the paths in `db.sh`. `DATASET_LIBRITTS_R` and `LIBRITTS` are mandatory;
`DATASET_EARS` and `DATASET_VCTK_DEMAND` supply the 48 kHz material. Any
`NOISE_*` variable that points at a real directory is added to the noise pool.

## Stages

| Stage | What |
|---|---|
| 1 | Data preparation (`local/data.sh`) |
| 2 | Resample the feature-predictor sets to 16 kHz |
| 3 | Pre-generate the RIR pool (needs `pyroomacoustics`) |
| 4 | Collect feature-predictor statistics |
| 5 | Train the feature predictor |
| 6 | Inference with the official Sidon vocoder (`--sidon_vocoder`) |
| 7 | Scoring: DNSMOS, NISQA, SpkSim, WER (`local/score.py`) |
| 8 | VERSA scoring, reference-free and reference-based |

```bash
./run.sh --stage 1 --stop_stage 5 --ngpu 4 --nj 64
./run.sh --stage 6 --stop_stage 8 --sidon_vocoder /path/to/decoder_cuda.pt
```

## Configs

`conf/train.yaml` and `conf/decode.yaml` are the defaults used by `run.sh`;
`conf/train.yaml` is a symlink to the variant under `conf/tuning/`.

## Notes

The RIR pool is generated ahead of training rather than simulated on the fly.
On-the-fly `pyroomacoustics` simulation is CPU-bound and starves the GPUs; with
a pre-generated pool the dataloader keeps 4x A40 at ~98% SM occupancy
(`iter_time` ~1e-4 s per step).
