# KoSP2E ASR Recipe

This is the ESPnet2 recipe for the **KoSP2E (Korean Speech Perception and Production Experiment)** dataset.

---

# 1. Overview

The **KoSP2E dataset** is a large-scale Korean speech corpus designed for speech perception and production experiments.
This recipe provides a full ASR pipeline using ESPnet2 with both Transformer and Conformer architectures.

## Hugginface Model Link
* https://huggingface.co/espnet/kosp2e-asr-ko

---

## 2. Dataset Preparation

### Step 1. Obtain the dataset

Contact the **KoSP2E dataset authors** to request access to:
- `data.zip` (contains audio files)
- `metadata.zip` (contains metadata spreadsheets)
    - You need to ask authors for metadata.

Once you receive them:
1. Download `data.zip` and unzip under the `wavs/` directory.
2. Download `metadata.zip` and unzip under the `downloads/metadata/` directory.

Your folder structure should look like this:

```
kosp2e/asr1/
├── conf/
├── data/
├── downloads/
│   ├── metadata/
│   │   ├── covid_train.xlsx
│   │   ├── covid_dev.xlsx
│   │   ├── covid_test.xlsx
│   │   ├── kss_train.xlsx
│   │   ├── kss_dev.xlsx
│   │   ├── kss_test.xlsx
│   │   ├── stylekqc_train.xlsx
│   │   ├── stylekqc_dev.xlsx
│   │   ├── stylekqc_test.xlsx
│   │   ├── zeroth_train.xlsx
│   │   ├── zeroth_dev.xlsx
│   │   └── zeroth_test.xlsx
├── local/
├── pyscripts/
├── scripts/
├── steps/
├── utils/
├── wavs/
│   ├── covid/
│   ├── kss/
│   ├── stylekqc/
│   └── zeroth/
├── setup_data.py
├── setup.sh
├── run.sh
└── asr.sh
```

---

# 2. Training

To start ASR training:

```
./run.sh
```

This script runs the standard ESPnet2 training pipeline with the Conformer model and BPE-2000 tokenization.

---

# 3. Results

Environment
* Date: Mon Nov 10 20:35:20 UTC 2025
* Python: 3.10.19
* ESPnet: 202509
* PyTorch: 2.9.0+cu128
* Model: Conformer (BPE=2000)
* Decode: Transformer LM (valid.acc.ave)

### WER
| dataset | Snt | Wrd  | Corr | Sub | Del | Ins | Err | S.Err |
|--------|----:|-----:|----:|---:|---:|---:|----:|-----:|
| test   | 2320 | 22337 | 77.1 | 20.4 | 2.6 | 4.4 | 27.4 | 76.4 |

### CER
| dataset | Snt | Wrd  | Corr | Sub | Del | Ins | Err | S.Err |
|--------|----:|-----:|----:|---:|---:|---:|----:|-----:|
| test   | 2320 | 84267 | 92.5 | 5.7 | 1.8 | 1.7 | 9.2  | 76.4 |

### TER
| dataset | Snt | Wrd  | Corr | Sub | Del | Ins | Err | S.Err |
|--------|----:|-----:|----:|---:|---:|---:|----:|-----:|
| test   | 2320 | 65361 | 89.4 | 8.6 | 2.0 | 2.1 | 12.7 | 76.4 |

---

# 5. References
* KoSP2E paper: https://arxiv.org/abs/2107.02875
---
