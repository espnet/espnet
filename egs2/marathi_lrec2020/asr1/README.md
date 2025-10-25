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

This recipe is for the **Marathi language** and is trained on the
[**IndicCorpora Marathi dataset**](https://www.cse.iitb.ac.in/~pjyothi/indiccorpora/#marathi).
All experiments use **Conformer architectures** with different tokenization schemes and frontends.

---

### Experiment Summary

| Model | Token | Epochs | Train Acc (%) | Val Loss (%) | **WER (%)** | **CER (%)** |
|:--|:--:|--:|--:|--:|--:|--:|
| **Char Conformer** | Char | 31 / 30 | 98.3 | 47.75 | **45.2** | **22.0** |
| **BPE-150 Conformer** | BPE-150 | 10 / 30 | 96.8 | 52.962 | **90.1** | **26.5** |
| **BPE-2000 Conformer** | BPE-2000 | 25 / 30 | 99.6 | 53.6 | **89.2** | **42.1** |
| **XLSR-Conformer (BPE-2000)** | BPE-2000 | 22 / 60 | 97.8 | 66.832 | **99.2** | **51.0** |
| **XLSR-Conformer (Char)** | Char | 13 / 60 | 82.4 | 75.155 | **78.7** | **43.3** |

---

### Notes
- The **Char Conformer** achieved the lowest validation loss (47.75) and best generalization.
- **BPE conformer** reached high training accuracy but tended to overfit.
- **XLSR + conformer (BPE and Char)** underperformed in this setup, likely due to limited fine-tuning (also sub-sampling conv2d was disabled. I used linear for this.)
All the above training was done without any LM model.

---

### Dataset Reference
> P. Jyothi et al., *“IndicCorpora: A Large Multilingual Corpus for Indic Languages.”*
> [IIT Bombay IndicCorpora Project – Marathi](https://www.cse.iitb.ac.in/~pjyothi/indiccorpora/#marathi)
