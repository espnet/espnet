---
tags:
- espnet
- speech-enhancement
- denoising
- bsrnn
- snoar
license: ${license}
library_name: espnet
pipeline_tag: audio-to-audio
---

# BSRNN speech denoiser trained on commercial-use URGENT 2025 data

A 16 kHz single-channel speech **denoiser** (BSRNN, 64 channels x 6 layers,
2 outputs: speech and noise, with energy mixture consistency). It is trained
on the subset of the URGENT 2025 challenge pool whose licenses allow
commercial use and derivatives without share-alike. It removes additive
noise and **keeps reverberation**: half of the training crops are
reverberant, and their target is the reverberant speech.

It was trained as the supervised seed of SNOAR self-adaptation (ESPnet3
`snoar` system), and can be adapted to a folder of unlabeled noisy
recordings with `espnet snoar adapt`.

- Repository: `${hf_repo}`
- Recipe: `${recipe}` (config `conf/training.yaml` in this repository)
- Git: `${git_head}` (${git_dirty})
- Created: `${created_at}`

${model_summary_section}

## Usage

```bash
pip install espnet
espnet snoar enhance noisy.wav -o enhanced.wav --model ${hf_repo}
# adapt to your own unlabeled recordings, starting from this model
espnet snoar adapt my_recordings/ -o my_enhancer/ --seed ${hf_repo}
```

```python
from espnet3.publication import InferenceModel

model = InferenceModel.from_pretrained("${hf_repo}")
enhanced = model({"speech": (16000, waveform)})["enhanced"].array
```

Input is resampled to 16 kHz mono. Recordings longer than 20 s are processed
in 20 s chunks with 50 % overlap.

## Training data

1.17 M speech files (about 2020 h) and 32.6 k noise files (80.7 h), mixed on
the fly at -5 to 20 dB SNR, half of them convolved with a room impulse
response; 100 k steps at batch 64 (3 s crops).

| Data | License | Used |
|---|---|---|
| CommonVoice (URGENT 2025 subset) | CC0 | all |
| DNS5 read speech (LibriVox) | CC BY 4.0 | all |
| LibriTTS, MLS, VCTK (URGENT 2025 subsets) | CC BY 4.0 | all |
| FSD50K noise | per clip | CC0 and CC BY clips only |
| FMA (no vocals) | per track | CC0, CC BY and public-domain tracks only |
| DNS5 Freesound noise | per clip | CC0 and CC BY clips with a known license only |
| MUSAN noise | public domain / CC BY 3.0 | all 929 files |
| URGENT 2025 simulated wind noise | generated | all |
| DNS5 RIRs (SLR26, SLR28) | Apache 2.0 | all |

Not used: EARS and WHAM! (CC BY-NC), the DNS5 AudioSet clips (YouTube audio),
and every noise clip that is non-commercial, no-derivatives, share-alike,
Sampling+ or of unknown license. The per-file license of every noise file
comes from the dataset metadata (FSD50K `clips_info`, FSDKaggle2018 metadata,
FMA `raw_tracks.csv`, MUSAN `LICENSE`).

`data/noise_attribution.tsv` lists every noise file used, with its license,
author, title and source URL, as CC BY attribution. The speech corpora are
credited above; please keep this attribution when you redistribute the model.

The license assessment was done per file from public metadata, in good faith;
it is not legal advice. Check that it fits your use.

## Evaluation

URGENT 2026 track-1 dev set (2200 recordings, resampled to 16 kHz). The dev
set also contains reverberation, bandwidth limitation and other distortions,
so a denoising-only model that keeps reverberation scores lower against the
dry clean reference than a full restoration model.

${results_section}

| Model | SI-SDR (dB) | PESQ-WB | STOI | UTMOS |
|---|---|---|---|---|
| Noisy input | 2.82 | 1.26 | 0.769 | 1.51 |
| **BSRNN, commercial-use data** (`training_supervised_commercial_bsrnn.yaml`) | **8.68** | **1.64** | **0.819** | **2.00** |
| BSRNN, whole pool (`training_supervised_bsrnn.yaml`, non-commercial data) | 9.01 | 1.65 | 0.825 | 2.02 |
| URGENT 2025 TF-GridNet (`kohei0209/tfgridnet_urgent25`, full restoration) | 11.73 | 2.32 | 0.907 | 2.37 |

UTMOS is UTMOS22 strong (SpeechMOS v1.2.0). The whole-pool model was trained
with non-commercial data and is not published; it shows what the license
restriction costs (0.3 dB SI-SDR).

## SNOAR self-adaptation

This model is meant as a seed: `espnet snoar adapt` fine-tunes it on a folder
of your own noisy recordings, with no clean references. How much that helps
was measured in the ESPnet3 WHAM! recipe (`egs3/wham/snoar`), with seeds
trained with supervision on Fisher (FastMSS) mixtures, so out of domain for
WHAM!. SNOAR adapted them on the unlabeled WHAM! `tr` recordings; scores are
on the separate `tt` split (3000 utterances, 16 kHz, `last.ckpt`, no
checkpoint selection on labels):

| Model | SI-SDR (dB) | PESQ-WB | STOI |
|---|---|---|---|
| TF-GridNet Fisher seed | 6.87 | 1.49 | 0.847 |
| SNOAR from it | **12.61** | **1.95** | **0.931** |
| BSRNN Fisher seed | 7.92 | 1.53 | 0.850 |
| SNOAR from it | **11.81** | **1.67** | **0.916** |

These are other seeds, not this model, and they are not published (Fisher is
LDC data). SNOAR adaptation starting from this model has not been measured
yet.

## Citation

```bibtex
@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and
    Jiro Nishitoba and Yuya Unno and Nelson Yalta and Jahn Heymann and Matthew Wiesner
    and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={{ESPnet}: End-to-End Speech Processing Toolkit},
  year={2018},
  booktitle={Proceedings of Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456}
}

@inproceedings{yu2023efficient,
  title={Efficient Monaural Speech Enhancement with Universal Sample Rate Band-Split RNN},
  author={Yu, Jianwei and Luo, Yi},
  booktitle={Proc. ICASSP},
  year={2023}
}
```
