# RESULTS

Overall results
| Model (conf name) | EER(%) | minDCF | Note | Huggingface |
|---|---|---|---|---|
| [conf/train_xvector.yaml](conf/train_xvector.yaml) | 1.81 | 0.1251 | | https://huggingface.co/espnet/voxcelebs12_xvector_mel |
| [conf/train_mfa_conformer.yaml](conf/train_mfa_conformer.yaml) | 0.782 | 0.0656 | | https://huggingface.co/espnet/voxcelebs12_mfaconformer_mel |
| [conf/train_ECAPA_mel.yaml](conf/train_ECAPA_mel.yaml) | 0.856 | 0.0666 | | https://huggingface.co/espnet/voxcelebs12_ecapa_mel |
| [conf/train_rawnet3.yaml](conf/train_rawnet3.yaml) | 0.739 | 0.0581 | | https://huggingface.co/espnet/voxcelebs12_rawnet3 |
| [conf/train_SKA_mel.yaml](conf/train_SKA_mel.yaml) | 0.729 | 0.0457 | | https://huggingface.co/espnet/voxcelebs12_ska_mel |
| [conf/train_ECAPA_wavlm_frozen.yaml](conf/train_ECAPA_wavlm_frozen.yaml) | 0.638 | 0.0499 | ECAPA-TDNN w/ Frozen WavLM | https://huggingface.co/espnet/voxcelebs12_ecapa_frozen|
| [conf/train_ECAPA_wavlm_joint.yaml](conf/train_ECAPA_wavlm_joint.yaml) | 0.394 | 0.0379 | ECAPA-TDNN w/ Jointly fine-tuned WavLM | https://huggingface.co/espnet/voxcelebs12_ecapa_wavlm_joint |
| [conf/train_SKA_wavlm_frozen.yaml](conf/train_SKA_wavlm_frozen.yaml) | 0.564 | 0.0548 | SKA-TDNN w/ Frozen WavLM | https://huggingface.co/espnet/voxcelebs12_ska_wavlm_frozen |
| [conf/train_SKA_wavlm_joint.yaml](conf/train_SKA_wavlm_joint.yaml) | 0.516 | 0.0429 | SKA-TDNN w/ Jointly fine-tuned WavLM | https://huggingface.co/espnet/voxcelebs12_ska_wavlm_joint |

## Environments - conf/train_rawnet3.yaml
date: 2023-11-21 14:02:42.314875

- python version: 3.9.16 (main, Mar  8 2023, 14:00:05)  [GCC 11.2.0]
- espnet version: 202310
- pytorch version: 2.0.1

| | Mean | Std |
|---|---|---|
| Target | -0.8015 | 0.1383 |
| Non-target | 0.0836 | 0.0836 |

| | EER(%) | minDCF |
|---|---|---|
|  rawnet3 | 0.739 | 0.05818 |%

## Environments - conf/train_ECAPA_mel.yaml
date: 2023-11-30 15:46:46.690435

- python version: 3.9.16 (main, May 15 2023, 23:46:34)  [GCC 11.2.0]
- espnet version: 202310
- pytorch version: 1.13.1

| | Mean | Std |
|---|---|---|
| Target | -0.8091 | 0.1398 |
| Non-target | 0.0795 | 0.0795 |

| Model name | EER(%) | minDCF |
|---|---|---|
| ecapa | 0.978 | 0.06860 |

## Environments - conf/train_mfa_conformer.yaml
date: 2023-11-30 15:48:13.707576

- python version: 3.9.16 (main, May 15 2023, 23:46:34)  [GCC 11.2.0]
- espnet version: 202310
- pytorch version: 1.13.1

| | Mean | Std |
|---|---|---|
| Target | -0.8149 | 0.1386 |
| Non-target | 0.0828 | 0.0828 |

| Model name | EER(%) | minDCF |
|---|---|---|
| mfa-conformer | 0.952 | 0.05834 |

## Environments - conf/train_ECAPA_wavlm_frozen.yaml
date: 2023-11-30 15:49:50.531573

- python version: 3.9.16 (main, May 15 2023, 23:46:34)  [GCC 11.2.0]
- espnet version: 202310
- pytorch version: 1.13.1

| | Mean | Std |
|---|---|---|
| Target | -0.7351 | 0.1314 |
| Non-target | 0.0959 | 0.0959 |

| Model name | EER(%) | minDCF |
|---|---|---|
| ecapa-frozenWavLM | 0.606 | 0.04467 |

## Environments - conf/conf/train_ECAPA_wavlm_joint.yaml
date: 2023-11-30 15:56:55.514299

- python version: 3.9.16 (main, Mar  8 2023, 14:00:05)  [GCC 11.2.0]
- espnet version: 202310
- pytorch version: 2.0.1

| | Mean | Std |
|---|---|---|
| Target | -0.7353 | 0.1212 |
| Non-target | 0.0916 | 0.0916 |

| Model name | EER(%) | minDCF |
|---|---|---|
| ecapa-jointWavLM | 0.425 | 0.04015 |
