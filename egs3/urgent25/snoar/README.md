# URGENT 2025 snoar recipe

Supervised 16 kHz denoising seeds for SNOAR self-adaptation (ESPnet3 `snoar`
system, see `egs3/wham/snoar`), trained on the URGENT 2025 challenge pool
with mixtures made on the fly. The models are BSRNN (64 channels x 6 layers)
with two outputs, speech and noise, and energy mixture consistency.

The seeds **denoise only**. Half of the training crops are convolved with a
room impulse response, and their speech target is the reverberant speech as it
is in the input, as mixture consistency (speech + noise = mixture) requires.
A seed that also removed reverberation would put the reverberation into its
noise estimate, and SNOAR's remixing would then learn from that.

Two training pools:

- `training_supervised_bsrnn.yaml` (default): the whole URGENT 2025 pool,
  including non-commercial data (EARS, WHAM! noise, NC clips of FSD50K and
  FMA). It needs no license metadata.
- `training_supervised_commercial_bsrnn.yaml` (option): only data whose
  license allows commercial use and derivatives without share-alike. Its
  `create_dataset` sets `commercial_only: true`, which runs a per-file
  license audit. This is the published model.

## Data Preparation

Training reads the URGENT 2025 training data from an OMNIIO store: one
directory per subset (`speech_commonvoice`, `noise_fsd50k`, `rir_dns5`, ...)
with a `metadata.parquet` (`id`, `sample_rate`, `samples`, `path`,
`start_byte`, `end_byte`) whose rows point at byte ranges of blob files.
Prepare the data with the challenge scripts
(<https://github.com/urgent-challenge>) and pack each subset that way.
Evaluation uses the URGENT 2026 track-1 dev set (`dev_noisy`, `dev_clean`)
from a store of the same layout.

The default pool needs only the two stores. The commercial-use pool also
needs the per-file license of every noise clip, which is read from the
dataset metadata, not from the audio:

| Variable | Content |
|---|---|
| `URGENT25_ROOT` | URGENT 2025 store |
| `URGENT26_ROOT` | URGENT 2026 store with `dev_noisy`, `dev_clean` |
| `URGENT_OUTPUT` | where `create_dataset` writes (default `./data`) |
| `FSD50K_META` (commercial only) | extracted `FSD50K.metadata.zip` (<https://zenodo.org/records/4060432>) |
| `FMA_META` (commercial only) | extracted `fma_metadata.zip` (<https://github.com/mdeff/fma>) |
| `FSDKAGGLE2018_META` (commercial only) | FSDKaggle2018 `meta/` (licenses of the DNS5 Freesound clips not in FSD50K) |
| `MUSAN_NOISE` (commercial only) | MUSAN `noise/` directory (extra public-domain / CC BY noise) |

`create_dataset` (`dataset/builder.py`) writes into `URGENT_OUTPUT` the dev
set and, with `commercial_only: true`, the license files:

| File | Content |
|---|---|
| `manifest/dev.jsonl`, `dev16k/` | the URGENT 2026 dev set resampled to 16 kHz |
| `noise_licenses.tsv` | license, author, title and URL of every noise file |
| `noise_attribution.tsv` | the same for the files used (CC0, CC BY, public domain): the CC BY credit |
| `commercial_allow.txt` | `<subset>/<id>` of the usable URGENT noise files |
| `musan_noise_commercial.txt` | the usable MUSAN files |

The audit can also be run alone:

```bash
python -m egs3.urgent25.snoar.dataset.licenses $URGENT_OUTPUT \
  --urgent_root $URGENT25_ROOT --fsd50k_meta $FSD50K_META --fma_meta $FMA_META \
  --fsdkaggle_meta $FSDKAGGLE2018_META --musan_noise $MUSAN_NOISE
```

The commercial-use pool:

| Data | License | Used |
|---|---|---|
| Speech: CommonVoice | CC0 | all |
| Speech: DNS5 read speech (LibriVox), LibriTTS, MLS, VCTK | CC BY 4.0 | all |
| Speech: EARS | CC BY-NC 4.0 | no |
| Noise: FSD50K | per clip | CC0 / CC BY clips (27.6 k of 32 k) |
| Noise: FMA no-vocal | per track | CC0 / CC BY / public domain (1460 of 19.9 k tracks) |
| Noise: DNS5 Freesound clips | per clip | CC0 / CC BY with a known license (1.4 k) |
| Noise: DNS5 AudioSet clips | YouTube audio | no |
| Noise: WHAM! | CC BY-NC 4.0 | no |
| Noise: URGENT simulated wind | generated | all |
| Noise: MUSAN (not in URGENT 2025) | public domain / CC BY 3.0 | 929 files |
| RIRs: DNS5 (SLR26, SLR28) | Apache 2.0 | all |

That is 1.17 M speech files (about 2020 h) and 32.6 k noise files (80.7 h),
against about 440 h of noise in the whole pool. The speech-corpus licenses are
those the corpora are distributed under; the noise licenses are per file from
the metadata above.

## Commands

Run from `egs3/urgent25/snoar` after `. ./path.sh`, with the variables above
set.

```bash
C=conf/training_supervised_bsrnn.yaml   # commercial-use data only: training_supervised_commercial_bsrnn.yaml

# 16 kHz dev set (and, for the commercial config, the license lists)
python run.py --stages create_dataset --training_config $C
# train (100 k steps, ~7 h on one H200) and export
python run.py --stages train export --training_config $C
# enhance the dev set and score SI-SDR / PESQ (wb) / STOI
python run.py --stages infer measure --training_config $C \
  --inference_config conf/inference.yaml --metrics_config conf/metrics.yaml
# pack and upload (commercial-use model only; data_dir must be under the
# recipe directory for the attribution file, e.g. a `data` symlink)
python run.py --stages pack_model upload_model --training_config $C \
  --inference_config conf/inference.yaml --metrics_config conf/metrics.yaml \
  --publication_config conf/publication.yaml
```

The exported model is the seed of SNOAR runs (`model.init`) and what
`espnet snoar enhance --model` and `espnet snoar adapt --seed` take.

## Results

URGENT 2026 track-1 dev (2200 recordings, resampled to 16 kHz), scored against
the dry clean reference. The dev set also has reverberation, bandwidth
limitation and other distortions, so a denoising-only seed scores lower there
than a full restoration model such as the URGENT 2025 TF-GridNet.

### BSRNN seeds

| Model | SI-SDR (dB) | PESQ-WB | STOI | UTMOS |
|---|---|---|---|---|
| Noisy input | 2.82 | 1.26 | 0.769 | 1.51 |
| **BSRNN, commercial-use data** (`training_supervised_commercial_bsrnn.yaml`) | **8.68** | **1.64** | **0.819** | **2.00** |
| BSRNN, whole pool (`training_supervised_bsrnn.yaml`, non-commercial data) | 9.01 | 1.65 | 0.825 | 2.02 |
| URGENT 2025 TF-GridNet (`kohei0209/tfgridnet_urgent25`, full restoration) | 11.73 | 2.32 | 0.907 | 2.37 |

SI-SDR, PESQ-WB and STOI are from `measure`; UTMOS (UTMOS22 strong,
SpeechMOS v1.2.0) was computed separately with the same enhanced outputs.
Both BSRNN models are the `last.ckpt` of 100 k steps (no checkpoint
selection). Restricting the pool to commercial-use data costs 0.3 dB SI-SDR;
most of the restriction falls on the noise (81 h instead of about 440 h).

Pretrained model (commercial-use data, CC BY 4.0):
<https://huggingface.co/espnet/urgent25_snoar_supervised_commercial_bsrnn>.
The whole-pool model is not published.
