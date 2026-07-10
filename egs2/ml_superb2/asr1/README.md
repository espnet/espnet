# ML-SUPERB 2.0 2024 Challenge


This is a recipe to reproduce the baseline model for the [Interspeech 2024 ML-SUPERB 2.0 Challenge](multilingual.superbbenchmark.org). While the challenge is open-ended, the organizers have provided here a minimal training and development set based off of the [ML-SUPERB 2.0 Benchmark](https://www.isca-archive.org/interspeech_2024/shi24g_interspeech.pdf) for participants to use. This data will cover most of the evaluated languages. More information about the challenge and the dataset construction can be found on the [challenge website](https://multilingual.superbbenchmark.org/challenge-interspeech2025/challenge_overview).


The baseline uses frozen SSL features from [MMS 1B](https://www.jmlr.org/papers/v25/23-1318.html), which are input into a 2-layer Transformer trained using CTC loss. It takes roughly 2 days to train on a single H100 GPU.
We recommend allocating at least 4 CPUs and at least 32GB of RAM. If GPU OOM occurs (such as when using a 40GB VRAM GPU), you can halve the batch size and double the gradiant accumulation.

## Refined LID-label Configs

The following configs use the refined LID-label data setting reported below:

- `conf/tuning/train_mms_baseline_refined.yaml`: 2-layer Transformer baseline.
- `conf/tuning/train_mms_ctc_transformer_lr1e-4.yaml`: 24-layer Transformer encoder.
- `conf/tuning/train_mms_ctc_conformer_12_macaron_lr1e-4.yaml`: 12-layer Conformer encoder with macaron-style feed-forward modules.
- `conf/tuning/train_mms_ctc_e_branchformer_12_nomacaron_lr1e-4.yaml`: 12-layer E-Branchformer encoder with `macaron_ffn: false`.

To prepare the refined LID-label data, run `local/data_refine.sh` instead of
the default `local/data.sh`.  In addition to the original data preparation
behavior, this refined script adds the following LID-label corrections:

- `lga -> lug`
- `ory -> ori`
- `arb -> ara`
- `nno`, `nob`, and `nor` are removed from the standard dev set.

For example:
```
./run.sh --asr_config conf/tuning/train_mms_baseline_refined.yaml
./run.sh --asr_config conf/tuning/train_mms_ctc_transformer_lr1e-4.yaml
./run.sh --asr_config conf/tuning/train_mms_ctc_conformer_12_macaron_lr1e-4.yaml
./run.sh --asr_config conf/tuning/train_mms_ctc_e_branchformer_12_nomacaron_lr1e-4.yaml
```

## Scoring

The challenge will use a custom scoring script, which considers worst language performance and CER standard deviation in addition to the typical multilingual ASR metrics of language identification accuracy and ASR CER. The exact implementation can be found in `local/score.py`, which also creates a `challenge_results.md` under your experimental directory with scores that correspond to the challenge system.

If you want to use the scoring script with a non-ESPnet, here is breakdown of the expected file format.
The script assumes that the reference and hypothesis are stored in a kaldi style text file:
```
uttid00 [langid0] this is a sample text
uttid01 [langid1] this is a sample text1
uttid02 [langid2] this is a sample text2
```
The script will look for the ref/hyp under this structure:
```
    root/
      - data/dev # kaldi style folder
            - text # ref file for standard dev set
      - data/dev_dialect # kaldi style folder
            - text # ref file for dialect dev set

      - exp/asr_train_asr_raw_char # your model folder
            - decode_asr_asr_model_valid.loss.ave # inference results
                - org/dev
                    - text # hyp file for standard dev set
                - dev_dialect
                    - text # hyp file for dialect dev set
            - challenge_results.md # generated results file
```
The script can be used directly with the following:
```
  python local/score.py --exp_dir <your model folder as shown above>
  # example
  python local/score.py --exp_dir exp/asr_train_asr_raw_char
```

You can also use the scoring functions `score()` and `score_dialect()`for the standard/dialect sets directly, which will normalize the text and calculate the challenge metrics for you.
Both functions have the following input parameters:
```
  references: list of reference text strings. Each string should be in format "[lid] asr reference text".
  lids: list of reference language id strings. Each string should be in format "[lid]" Note that we always assume a 3-character ISO string for the LID target.
  hyps: list of hypothesis text strings. Each string should be in format "[lid] asr reference text".
```
Such that `references[i]`, `lids[i]`, and `hyps[i]` should all correspond to the respective data for the ith utterance.
```
  from local.score import score, score_dialect
  reference = ["[eng] here is a sample reference", "[fra] another utterance"]
  reference_lid = ["[eng]", "[fra]"]
  hypothesis = ["[eng] here is a sample reference from asr", "[eng] another one"]
  lid_result, cer_result, worst_result, std_result = score(reference, reference_lid, hypothesis)
  lid_dialect, cer_dialect, = score_dialect(reference, reference_lid, hypothesis)
```

## RESULTS

### train_asr.yaml (Frozen MMS 1B + Transformer + CTC)

### Environments
- date: `Sat Dec 28 11:08:07 CST 2024`
- python version: `3.10.15 (main, Oct  3 2024, 07:21:53) [GCC 11.2.0]`
- espnet version: `espnet 202409`
- pytorch version: `pytorch 2.6.0.dev20241008+cu124`
- model_link: https://huggingface.co/espnet/mms_1b_mlsuperb
- Git hash: `4fe2783ef85c294af19f36fb519ec62dc6639ce7`
  - Commit date: `Fri Dec 27 14:11:37 2024 +0000`

|decode_dir|Standard CER|Standard LID|Worst 15 CER|CER StD|Dialect CER|Dialect LID|
|---|---|---|---|---|---|---|
decode_asr_asr_model_valid.loss.ave|24.0|74.0|71.0|25.5|32.7|54.0|

### Refined LID-label data results

The following results were obtained after preparing the public ML-SUPERB 2.0
data with `local/data_refine.sh`.  The experiments use frozen MMS 1B features
and CTC training.  Decoding uses `valid.loss.ave_2best`.

Training settings:

- `batch_size: 8`
- `accum_grad: 4`
- `num_iters_per_epoch: 20000`
- `max_epoch: 20`

|model_name|decode_dir|Standard CER|Standard LID|Worst 15 CER|CER StD|Dialect CER|Dialect LID|
|---|---|---:|---:|---:|---:|---:|---:|
|Baseline 2-layer Transformer|decode_asr_asr_model_valid.loss.ave_2best|22.6|76.3|55.7|14.6|33.9|59.3|
|Transformer 24-layer|decode_asr_asr_model_valid.loss.ave_2best|19.5|81.3|52.3|14.4|36.2|63.1|
|Conformer 12-layer macaron|decode_asr_asr_model_valid.loss.ave_2best|21.3|72.5|57.9|16.0|39.6|62.0|
|E-Branchformer 12-layer no-macaron|decode_asr_asr_model_valid.loss.ave_2best|18.6|81.8|51.1|14.4|33.5|72.3|
