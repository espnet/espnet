# ESPnet-Sidon: feature-predictor speech restoration

An ESPnet reproduction of **Sidon** ([arXiv:2509.17052](https://arxiv.org/abs/2509.17052)).
The model predicts the clean SSL hidden state from degraded speech (stage 1
of the paper) and vocodes it to 48 kHz with a DAC-style decoder that is
either trained here (paper stages 2-3: pretrain on ground-truth features,
finetune on predicted ones) or taken from the official release.

This directory is `rst1` (restoration) rather than `enh1`: the pipeline is not
an `EnhancementTask` model and does not go through the standard `enh.sh`
driver. The feature predictor is trained by `espnet2.bin.rst_train`
(`RestorationTask`), which scores predicted SSL features rather than
waveforms, and the vocoder by `espnet2.bin.rst_vocoder_train`
(`RestorationVocoderTask`); `run.sh` drives the 11 stages directly because
the vocoder pretrain/finetune stages sit in the middle of the pipeline.

Two SSL backbones are supported (`ssl_encoder` in the config), both 1024-d at
50 Hz so the vocoder stages are identical:

| `ssl_encoder` | backbone | layer | weights | licence |
|---|---|---|---|---|
| `w2v_bert2` (default, paper) | w2v-BERT 2.0 | 8 | `facebook/w2v-bert-2.0` | MIT |
| `xeus` | XEUS (ESPnet E-Branchformer SSL, [Chen et al. 2024](https://arxiv.org/abs/2407.00837)) | block 10 | `espnet/xeus`, loaded with `SSLTask.build_model_from_file` | **CC-BY-NC-SA-4.0** (non-commercial) |

Select XEUS with `--config conf/tuning/train_sidon_xeus.yaml` for stage 5 and
pass the same `ssl_encoder` / `ssl_encoder_conf` to the vocoder configs (stages
7-8); inference reads the encoder type from the training config.

## Extra dependencies

These are **not** installed by `tools/` and are not ESPnet dependencies:

| Package | Needed by | Install |
|---|---|---|
| `peft` | LoRA adapters on the SSL student (stages 4-5) | `pip install peft` |
| `pyroomacoustics` | RIR pool generation (stage 3) | `pip install pyroomacoustics` |
| `versa` | VERSA scoring (stage 11) | `tools/installers/install_versa.sh` |
| NISQA *(optional)* | `local/score.py --nisqa_model` | clone [NISQA](https://github.com/gabrielmittag/NISQA), add to `PYTHONPATH` |

`transformers` supplies the w2v-BERT 2.0 backbone, `WavLMForXVector` for
speaker similarity, and `facebook/mms-1b-all` for WER. All three are fetched
from the Hub on first use, so stages 4-10 need network access (or a pre-warmed
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
| 6 | Collect vocoder statistics (48 kHz sets) |
| 7 | Pretrain the vocoder on ground-truth SSL features of clean speech |
| 8 | Finetune the vocoder on the stage-5 predictor's features of degraded speech |
| 9 | Inference with the stage-8 vocoder, or the official one (`--sidon_vocoder`) |
| 10 | Paper's four metrics: DNSMOS, NISQA, SpkSim, WER (`local/score.py`, dependency-light) |
| 11 | VERSA scoring, reference-free and reference-based (recommended: same metrics plus UTMOS, SQUIM, PESQ, STOI, SDR/SI-SNR and more in one pass) |

```bash
./run.sh --stage 1 --stop_stage 8 --ngpu 4 --nj 64     # predictor + vocoder
./run.sh --stage 9 --stop_stage 11                      # uses exp/sidon_vocoder_finetune
# or skip vocoder training and use the official decoder
./run.sh --stage 9 --stop_stage 11 --sidon_vocoder /path/to/decoder_cuda.pt
```

## Vocoder

Stages 7-8 train the same DAC decoder as the official release (52.4M
parameters, strides 8-5-4-3-2 = 960x, one 20 ms feature frame to 960 samples
at 48 kHz) with ESPnet's `GANTrainer`, the multi-period + multi-band STFT
discriminator from `espnet2.gan_codec` and the mel / adversarial / feature
matching losses from `espnet2.gan_tts.hifigan` (weights 15 / 2 / 1, summed over
sub-discriminators as in DAC). The encoder is frozen in both stages; the
generator and discriminators run on a 1 s excerpt whose features were computed
with 8 s of context (`segment_duration`, `context_duration`).

Two vocoders share the stage-7/8 data path and the inference loader; the
config's `vocoder_type` selects one and `vocoder_conf` configures it.

| `vocoder_type` | Model | Training | Config (stage 7) |
|---|---|---|---|
| `dac` (default) | DAC decoder as in the official release, 52.4M | GAN, `rst_vocoder_train` | `train_sidon_vocoder_pretrain.yaml` |
| `hifigan` | ESPnet `HiFiGANGenerator`, 512 channels, 17M | GAN, `rst_vocoder_train` | `train_sidon_vocoder_pretrain_hifigan.yaml` |

The official vocoder is published only as a frozen TorchScript graph.
`local/convert_official_sidon_vocoder.py` recovers its weights into the
recipe's module (verified bit-exact against the graph), which lets stage 8
start from the published vocoder instead of a stage-7 run:

```bash
python local/convert_official_sidon_vocoder.py \
    --torchscript /path/to/decoder_cuda.pt --out_dir exp/official_sidon_vocoder
./run.sh --stage 8 --stop_stage 8 \
    --vocoder_init exp/official_sidon_vocoder/vocoder.pth --discriminator_init ""
```

## Configs

`conf/train.yaml` and `conf/decode.yaml` are the defaults used by `run.sh`;
`conf/train.yaml` is a symlink to the variant under `conf/tuning/`.

## Notes

The RIR pool is generated ahead of training rather than simulated on the fly.
On-the-fly `pyroomacoustics` simulation is CPU-bound and starves the GPUs; with
a pre-generated pool the dataloader keeps 4x A40 at ~98% SM occupancy
(`iter_time` ~1e-4 s per step).
