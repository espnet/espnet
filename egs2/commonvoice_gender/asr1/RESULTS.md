# RESULTS

## Environments

- date: `Thu Feb 19 09:39:46 EST 2026`
- python version: `3.12.8 | Intel Corporation | (main, Mar 31 2025, 23:28:18) [GCC 14.2.0]`
- espnet2 version: `espnet2 202511`
- pytorch version: `pytorch 2.7.1+cu126`
- Git hash: `239f5166a520566b132c359fb01ce38972aeeefc`
  - Commit date: `Thu Feb 5 04:31:11 2026 -0800`

## Model Information

- **Architecture**: Conformer encoder + Transformer decoder
- **Training data**: ~100 hours of CommonVoice English (gender-filtered subsets)
- **Tokenization**: BPE with 150 subword units
- **Speed perturbation**: 0.9x, 1.0x, 1.1x
- **Language Model**: None (no LM used)

## Experimental Results

### Same-Gender Evaluation (In-Gender Performance)

#### Male-Trained Model on Male Speech

| Dataset | Snt | Wrd | WER | CER | TER |
|---------|-----|-----|-----|-----|-----|
| test_male_en | 1753 | 16161 | **32.0%** | 14.5% | 20.9% |
| dev_male_en | 1891 | 18239 | **30.1%** | 12.8% | 19.1% |

#### Female-Trained Model on Female Speech

| Dataset | Snt | Wrd | WER | CER | TER |
|---------|-----|-----|-----|-----|-----|
| test_female_en | 534 | 5151 | **33.1%** | 15.1% | 21.2% |
| dev_female_en | 453 | 4372 | **29.1%** | 12.1% | 17.3% |

### Cross-Gender Evaluation (Gender Bias Analysis)

#### Male-Trained Model on Female Speech

| Dataset | Snt | Wrd | WER | CER | TER |
|---------|-----|-----|-----|-----|-----|
| test_female_en | 534 | 5151 | **54.6%** | 19.4% | 23.1% |

**Bias Analysis**: WER increases from 32.0% (male→male) to **54.6%** (male→female), representing a **+22.6 percentage point increase** (~70% relative degradation).

#### Female-Trained Model on Male Speech

| Dataset | Snt | Wrd | WER | CER | TER |
|---------|-----|-----|-----|-----|-----|
| test_male_en | 1753 | 16161 | **37.0%** | 17.4% | 24.2% |

**Bias Analysis**: WER increases from 33.1% (female→female) to **37.0%** (female→male), representing a **+3.9 percentage point increase** (~12% relative degradation).

## Summary

This recipe demonstrates **gender-based ASR fairness analysis** using CommonVoice English data:

1. **Same-gender performance**: Both male and female models achieve similar WER (~32-33%) on their respective gender's speech, indicating balanced training data quality.

2. **Cross-gender bias**: 
   - **Male model shows significant bias** against female speech (54.6% WER vs 32.0% baseline)
   - **Female model shows moderate bias** against male speech (37.0% WER vs 33.1% baseline)
   - The asymmetry suggests that models trained on male speech generalize poorly to female speech, highlighting fairness concerns in ASR systems.

3. **Dataset**: 
   - Training: ~71.7K utterances per gender (~100 hours)
   - Male test: 1,753 utterances
   - Female test: 534 utterances

## Model Files

Pre-trained models are available in:
- Male model: `exp/asr_train_asr_conformer5_raw_en_bpe150_sp/`
- Female model: `exp/asr_train_asr_conformer5_raw_en_bpe150_female_sp/`
