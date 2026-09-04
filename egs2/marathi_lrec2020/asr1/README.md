# RESULTS

## Environment
- **Date:** `Sat Oct 25 08:19:46 UTC 2025`
- **Python version:** `3.9.23`
- **ESPnet version:** `202509`
- **PyTorch version:** `2.3.0+cu121`
- **CUDA version:** `12.1`
- **ESPnet Git hash (upstream):** `53e09761cb164b28f299e178262bf2056d8059d7`
  - **Commit date:** `Fri Oct 24 11:26:46 2025 +0900`

---

## Marathi ASR — `marathi_lrec2020`

Recipe for **Marathi** ASR on the
[**IndicCorpora Marathi subset**](https://www.cse.iitb.ac.in/~pjyothi/indiccorpora/#marathi).

Training uses **`conf/train_asr_transformer.yaml`** (character Conformer: 3 blocks, 256-dim encoder, `batch_bins: 16000000`, `accum_grad: 4`, Adam `lr: 0.0005`, warmup 20k, SpecAugment, hybrid CTC/attention `ctc_weight: 0.3`).

Decoding without LM: **`conf/decode_asr.yaml`** (`lm_weight: 0.0`).
Decoding with LM (match reported fusion): **`conf/decode_asr_lm.yaml`** (beam 20, `ctc_weight: 0.5`, `lm_weight: 0.3`).

---

### Test-set decoding (`marathi_test`)

Beam **20**, **CTC weight 0.5** unless noted.

#### Beam 20, CTC 0.5 (no LM)

|        | Corr | Sub | Del | Ins | Err | S.Err |
|--------|-----:|----:|----:|----:|----:|------:|
| **CER** | 88.9 | 7.1 | 4.0 | 1.9 | 13.0 | 77.7 |
| **WER** | 73.8 | 23.8 | 2.4 | 3.2 | 29.4 | 78.5 |

#### Beam 20, CTC 0.5, LM weight 0.3

|        | Corr | Sub | Del | Ins | Err | S.Err |
|--------|-----:|----:|----:|----:|----:|------:|
| **CER** | 89.0 | 6.6 | 4.4 | 1.7 | 12.6 | 74.3 |
| **WER** | 76.0 | 21.6 | 2.4 | 3.0 | 27.0 | 75.0 |

---

### Dataset reference

> P. Jyothi et al., *“IndicCorpora: A Large Multilingual Corpus for Indic Languages.”*
> [IIT Bombay IndicCorpora — Marathi](https://www.cse.iitb.ac.in/~pjyothi/indiccorpora/#marathi)
