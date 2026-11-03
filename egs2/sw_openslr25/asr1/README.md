# ALFFA Swahili Broadcast News ASR (OpenSLR SLR25)

## Corpus

Swahili broadcast-news portion of the ALFFA (African Languages in the Field:
speech Fundamentals and Automation) corpus: https://www.openslr.org/25/

- License: MIT
- Train: 9,546 utterances (~9 h) after holding out `sw_dev`
- Dev (`sw_dev`): 634 utterances, carved out of the shipped train set
- Test (`sw_test`): 1,991 utterances (~1.8 h), the official test set

The corpus ships only train/test splits, so `sw_dev` holds out two whole
recording sessions (`SWH-05-20110327`, `SWH-15-20101109`, one per show family)
rather than individual utterances, so no recording leaks between train and dev.
See `local/data_prep.py`.

```
@InProceedings{gelas:hal-00954048,
  author    = {Gelas, Hadrien and Besacier, Laurent and Pellegrino, Francois},
  title     = {Developments of {S}wahili resources for an automatic speech recognition system},
  booktitle = {SLTU - Workshop on Spoken Language Technologies for Under-Resourced Languages},
  year      = {2012},
  address   = {Cape Town, South Africa},
}
```

## Usage

```bash
./run.sh
```

The corpus is downloaded automatically into `downloads/`. Defaults: 1 GPU,
speed perturbation (0.9/1.0/1.1), BPE 300, no language model.

## Notes

- The tarball's `wav.scp` contains placeholder paths (`/my_dir/wav/...`).
  `local/data_prep.py` indexes every `.wav` under the download directory and
  matches utterances by file name instead of trusting those paths.
- Transcripts and session IDs come from the Kaldi-format `data/train` and
  `data/test` files shipped inside the corpus tarball.
- `egs2/iwslt21_low_resource` also uses ALFFA Swahili, but as one part of a
  combined training set evaluated on the IWSLT 2021 test sets. This recipe
  trains on ALFFA alone and reports results on ALFFA's official test set.

## Results

### Environments
- date: `Thu Oct 15 04:57:52 EDT 2026`
- python version: `3.12.8`
- espnet2 version: `espnet2 202610.post2`
- pytorch version: `pytorch 2.13.0+cu126`
- Git hash: `9dea2da7499f65344b40f140a1bb1bf7e263ca0b`

### asr_train_raw_bpe300_sp

Transformer encoder-decoder (6 encoder / 3 decoder blocks, 256-dim attention),
hybrid CTC/attention (CTC weight 0.3 in training, 0.5 in decoding),
SpecAugment, speed perturbation, BPE 300, average of the 10 best checkpoints,
no language model. About 2 hours on one V100 (32 GB).

#### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.acc.ave/sw_dev|634|5877|76.2|19.4|4.5|3.8|27.7|82.2|
|decode_asr_model_valid.acc.ave/sw_test|1991|17700|71.2|23.2|5.6|2.7|31.5|86.4|

#### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.acc.ave/sw_dev|634|37237|93.2|2.5|4.3|3.0|9.8|82.2|
|decode_asr_model_valid.acc.ave/sw_test|1991|111615|91.4|3.3|5.3|2.2|10.8|86.4|

#### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.acc.ave/sw_dev|634|15452|84.2|8.7|7.1|3.0|18.8|82.2|
|decode_asr_model_valid.acc.ave/sw_test|1991|45854|80.9|10.9|8.2|2.4|21.5|86.4|

### Comparison with published baselines

Kaldi GMM/SGMM results on the same test set, from
https://github.com/besacier/ALFFA_PUBLIC (these use a word-level n-gram
language model; this recipe uses none):

|System|Test WER (%)|
|---|---|
|Triphone (13 MFCC)|33.55|
|Triphone + LDA + MLLT|31.92|
|Triphone + LDA + MLLT + SAT + fMLLR|31.56|
|Triphone + LDA + MLLT + SAT + fMLLR + MMI/fMMI|30.87|
|Triphone + LDA + MLLT + SGMM|27.36|
|**This recipe (no LM)**|**31.5**|
