# MS-SNSD Speech Enhancement Recipe (enh1)

This recipe provides a speech enhancement (SE) baseline on the MS-SNSD dataset using TF-GridNet.

---

## 1. Dataset: MS-SNSD

- Official repository: https://github.com/microsoft/MS-SNSD
- License: MIT License (see the official repository)

MS-SNSD provides:
- train_clean
- train_noise
- test_clean
- test_noise

No fixed noisy waveforms are distributed. Noisy speech is generated following the official MS-SNSD mixing protocol.

---

## 2. Data Preparation

### Step 1: Download MS-SNSD

Clone the official repository:

```bash
git clone https://github.com/microsoft/MS-SNSD.git
```

Set the dataset path in db.sh:

```bash
MS_SNSD=/path/to/MS-SNSD
```

---

### Step 2: Generate Noisy Data

Noisy speech is generated using:

```bash
local/ms_snsd_create_mixture.sh
```

You can control the synthesized dataset size in:

local/ms_snsd_create_mixture.sh

```
total_hours_train=30
total_hours_test=1
```

The mixture generation follows the official MS-SNSD protocol.

A minimal fix is included in:

local/ms_snsd_noisy_speech_synthesizer.py

This ensures stable execution due to minor configuration parsing issues.

---

### Step 3: Prepare ESPnet Data Directories (train/valid/test)

Data preparation is performed in:

local/ms_snsd_data_prep.sh

This script:
- creates data/{tr,cv,tt}_ms_snsd
- performs train/valid split with ratio 8:2 (train:valid)
- uses a deterministic split

Final dataset names:

```
train_set=tr_ms_snsd
valid_set=cv_ms_snsd
test_sets=tt_ms_snsd
```

---

## 3. Training Configuration

Model:
- TF-GridNet

Config file:

./conf/tuning/train_enh_tfgridnet.yaml

Run training:

```bash
./run.sh
```

---

## 4. Experimental Results

### Environment

- date: Mon Feb 16 11:21:38 KST 2026
- python: 3.10.14
- espnet2: 202511
- pytorch: 2.10.0+cu128
- Git hash: 239f5166a520566b132c359fb01ce38972aeeefc

### Results (TF-GridNet)

- config: ./conf/tuning/train_enh_tfgrid.yaml
- model: https://huggingface.co/espnet/ms_snsd_tfgridnet

| dataset             | STOI  | SAR   | SDR   | SIR | SI_SNR |
|---------------------|-------|-------|-------|-----|--------|
| enhanced_cv_ms_snsd | 88.49 | 20.16 | 20.16 | 0.00 | 19.90  |
| enhanced_tt_ms_snsd | 96.01 | 17.84 | 17.84 | 0.00 | 17.52  |

Notes:
- All metrics are higher-is-better.

---

## 5. Additional Notes

- No fixed noisy test set is provided by MS-SNSD.
- Noisy signals are synthesized following the official mixing framework.
- Dataset size can be adjusted via total_hours_train and total_hours_test.
