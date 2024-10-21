# OWSM-CTC v3.1

[OWSM-CTC](https://aclanthology.org/2024.acl-long.549/) is an encoder-only speech foundation model based on hierarchical multi-task self-conditioned CTC.
This version is trained on 180k hours of public audio data for multilingual speech recognition, any-to-any speech translation, and language identification, which follows the design of the project, [Open Whisper-style Speech Model (OWSM)](https://arxiv.org/abs/2401.16658).

## Data Preparation

The training data follows the same format as the encoder-decoder OWSM v3.1, except that timestamps are removed from the `text` file. Please first follow the `egs2/owsm_v3.1/s2t1` recipe to prepare OWSM data, and then convert `text` into the new format by running `python local/convert_owsm_data.py` (the path to the BPE tokenizer needs to be modified to your path).

## Pre-trained Model

The pre-trained model is available at: https://huggingface.co/pyf98/owsm_ctc_v3.1_1B

The model page also contains example usage.
