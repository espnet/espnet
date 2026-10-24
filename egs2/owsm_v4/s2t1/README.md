# OWSM v4

The [Open Whisper-style Speech Model (OWSM)](https://www.wavlab.org/activities/2024/owsm/) project develops fully open speech foundation models using publicly available data and open-source toolkits.

OWSM v4 significantly outperforms previous versions in multilingual tasks. It is trained on a [clean version of YODAS](https://huggingface.co/datasets/espnet/yodas_owsmv4) along with previous OWSM data. Please refer to our paper for more details about the training process: https://arxiv.org/abs/2506.00338

Note: OWSM v4 applies 8 times subsampling (instead of 4 times in OWSM v3.1) to the log Mel features, leading to a final resolution of 80 ms in the encoder.
When running inference, keep `maxlenratio` at its default of `0.0` and avoid smaller positive values such as `0.5`, which can truncate the output. `0.0` gives the same maximum output length as `1.0` (the number of encoder frames), and it also enables the end detection of the beam search, so decoding stops early once the search has converged instead of running to the maximum length.

## Results

Please refer to our paper for comprehensive evaluations. Below are some notable results.

### Language Identification

![Language identification accuracy](local/lid-result.png)


### English ASR

![English ASR WER vs inference speed](local/en-asr.png)

### Multilingual ASR

![Multilingual ASR WER](local/fleurs.png)


## Data Cleaning

As presented in Section 2.1 of [our paper](https://arxiv.org/abs/2506.00338), we conducted three-stage data cleaning from the original [YODAS2](https://huggingface.co/datasets/espnet/yodas2) dataset.
- Resegmentation (Section 2.1.1 in the paper)
- LID-based filtering (Section 2.1.2)
- CTC-score-based filtering (Section 2.1.3)

To get started, download the YODAS2 dataset to a local directory and create a text file containing paths to all data files that need to be processed. For example, we create `data_reseg/json_files.txt`:
```
/work/hdd/bbjs/shared/corpora/yodas2/data/af000/text/00000000.json
/work/hdd/bbjs/shared/corpora/yodas2/data/af000/text/00000001.json
/work/hdd/bbjs/shared/corpora/yodas2/data/am000/text/00000000.json
/work/hdd/bbjs/shared/corpora/yodas2/data/am000/text/00000001.json
...
```

Then, pass the file list to `local/data.sh` which will filter the data and convert the processed version in Kaldi style for later training.

### Stage 1: Resegmentation

YODAS provides unsegmented long-form recordings, each of them is accompanied by a list of text transcriptions annotated with start and end timestamps. However, these timestamps can be inaccurate. Consequently, our first step is to realign the audio and text using the CTC segmentation algorithm.

We first launch parallel jobs for CTC segmentation in `local/ctc_seg.py`, and then resegment the short utterances into long-form utterances up to 30 seconds in `local/get_longform_from_reseg.py`. This makes the training data consistent with the Whisper-style training.

### Stage 2: LID-based filtering

We observe that certain utterances have incorrect language labels. To address this issue, we perform language identification on both audio and text using public models using `local/lid.py`. Then, we remove utterances where the language label does not match the identified language from either audio or text, as implemented in `local/filter_lid.py`.

### Stage 3: CTC-score-based filtering

The CTC segmentation algorithm assigns a score to each utterance, which indicates the confidence of the segmentation. We filter out utterances with low CTC scores using `local/filter_score.py`. The CTC confidence score is language-dependent; therefore, we rank the scores of short utterances within each language and select a relative threshold (quantile).

Finally, we convert the filtered data into Kaldi format using `local/convert_to_kaldi.py`. The resulting data is stored in `data`.

## Open ASR Leaderboard evaluation and CPU decoding speed

Open ASR Leaderboard test sets can be prepared and evaluated directly through the recipe stages (`local/data.sh`, `s2t.sh` stage 12 and stage 13):

### 1. Data Preparation (`local/data.sh`)

`local/data.sh` prepares the [Open ASR Leaderboard](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard) test sets using `local/prepare_hf_asr_leaderboard.py`. Two profiles are supported:

- **Full evaluation (`--openasr_subset full`, default)**: Prepares the full test sets (`data/openasr_<set>`) across benchmark configs (`librispeech` test.clean and test.other, `ami_cleaned`, `voxpopuli_cleaned_aa`, `common_voice`, `tedlium`, `gigaspeech_cleaned`, `spgispeech`, `earnings22`), preserving all utterances without sampling. Note that `--openasr_subset full` also creates the seeded 100-sample sets (`data/openasr_<set>_100`) alongside the full sets.
- **Small smoke test (`--openasr_subset small`)**: Prepares only the 100-utterance subsets (`data/openasr_<set>_100`) from 6 benchmark sets with reproducible seed 0 (`--max_dur 30.0`) for quick local validation on CPU/laptop in a few minutes.

The dataset is gated: accept the terms on its page and log in with `huggingface-cli login` (or set `HF_TOKEN`) before running.

```bash
# Prepare full leaderboard test sets (also creates the _100 subsets)
local/data.sh --stage 5 --stop_stage 5 --openasr_subset full

# Or prepare only the 100-utterance sample sets for quick smoke testing
local/data.sh --stage 5 --stop_stage 5 --openasr_subset small
```

### 2. Feature Dump (Stage 3), Decoding (Stage 12), and Scoring (Stage 13)

Format test sets, decode, and score using standard `s2t.sh` stages with the Open ASR decoding configuration (`conf/decode_openasr.yaml`: beam 5, ctc 0, maxlenratio 0, lang `<eng>`, task `<asr>`):

```bash
# Evaluate OWSM v4 base on the 100-utterance smoke test sets
./s2t.sh --stage 3 --stop_stage 13 --skip_train true \
    --download_model espnet/owsm_v4_base_102M \
    --test_sets "openasr_librispeech_test_clean_100 openasr_librispeech_test_other_100 openasr_ami_cleaned_test_100 openasr_voxpopuli_cleaned_aa_test_100 openasr_earnings22_test_100 openasr_common_voice_test_100" \
    --inference_config conf/decode_openasr.yaml \
    --gpu_inference false \
    --batch_size 8 \
    --inference_nj 4 \
    --cleaner whisper_en \
    --hyp_cleaner whisper_en

# Full leaderboard evaluation across all sets
./s2t.sh --stage 3 --stop_stage 13 --skip_train true \
    --download_model espnet/owsm_v4_base_102M \
    --test_sets "openasr_librispeech_test_clean openasr_librispeech_test_other openasr_ami_cleaned_test openasr_voxpopuli_cleaned_aa_test openasr_common_voice_test openasr_tedlium_test openasr_gigaspeech_cleaned_test openasr_spgispeech_test openasr_earnings22_test" \
    --inference_config conf/decode_openasr.yaml \
    --gpu_inference false \
    --batch_size 8 \
    --inference_nj 8 \
    --cleaner whisper_en \
    --hyp_cleaner whisper_en
```

To calculate RTF and latency over stage 12 decoding logs:

```bash
python3 pyscripts/utils/calculate_rtf.py \
    --log-dir "exp/espnet/owsm_v4_base_102M/decode_openasr/openasr_librispeech_test_clean_100/logdir" \
    --log-name "s2t_inference" \
    --input-shift 0.0625 \
    --start-times-marker "speech length" \
    --end-times-marker "best hypo"
```

### Results and Observations

See [RESULTS.md](RESULTS.md) for detailed numbers and findings, including:
- Sample smoke test benchmark results across decoding setups (beam 5, batch 8, greedy).
- Analysis of end detection behavior and `maxlenratio=0` speedup on padded 30-second single-window inputs.
- Discussion of 370M model behavior on AMI and greedy search repetition loops.

## OWSM series

### Encoder-decoder OWSM

| Name | Size | Hugging Face Repo |
| :--- | ---: | :---------------- |
| OWSM v3.1 base | 101M | https://huggingface.co/espnet/owsm_v3.1_ebf_base |
| OWSM v3.1 small | 367M | https://huggingface.co/espnet/owsm_v3.1_ebf_small |
| OWSM v3.1 medium | 1.02B | https://huggingface.co/espnet/owsm_v3.1_ebf |
| OWSM v3.2 small | 367M | https://huggingface.co/espnet/owsm_v3.2 |
| OWSM v4 base | 102M | https://huggingface.co/espnet/owsm_v4_base_102M |
| OWSM v4 small | 370M | https://huggingface.co/espnet/owsm_v4_small_370M |
| OWSM v4 medium | 1.02B | https://huggingface.co/espnet/owsm_v4_medium_1B |


### CTC-based OWSM

| Name | Size | Hugging Face Repo |
| :--- | ---: | :---------------- |
| OWSM-CTC v3.1 medium | 1.01B | https://huggingface.co/espnet/owsm_ctc_v3.1_1B |
| OWSM-CTC v3.2 medium | 1.01B | https://huggingface.co/espnet/owsm_ctc_v3.2_ft_1B |
| OWSM-CTC v4 medium | 1.01B | https://huggingface.co/espnet/owsm_ctc_v4_1B |


### vLLM inference

The [ESPnet vLLM fork](https://github.com/espnet/vllm) runs encoder-decoder OWSM with ESPnet's own frontend and encoder and a vLLM decoder with paged KV caches. Export a native checkpoint and run it by following the fork's [OWSM guide](https://github.com/espnet/vllm/blob/main/examples/espnet/OWSM.md). Before you use it, note these limits from that guide:

- **Checkpoints.** The fork checks nine public checkpoints on H100 in float32 with `TRITON_ATTN`: v1, v2, v2 E-Branchformer, v3, v3.1 E-Branchformer small, v3.2, and v4 base/small/medium. BF16 with FlashAttention has not matched native tokens and log-probabilities. CTC-based OWSM is a different architecture, and the fork rejects it.
- **Decoding.** The fork supports greedy autoregressive decoding and sampling, but it does not implement hybrid CTC/attention beam search, external language-model fusion, timestamp pairing filters, or long-form segmentation. It is therefore not a drop-in replacement for every native decoding configuration. Its reported greedy-token agreement under the validated settings does not establish equivalence to the decoding settings used for the results above.
- **Input length.** By default, clips are padded to the 30-second training window, and longer clips are rejected. Split long recordings first.


### Citations

#### OWSM v4

```BibTex
@inproceedings{owsm-v4,
  title={{OWSM} v4: Improving Open Whisper-Style Speech Models via Data Scaling and Cleaning},
  author={Yifan Peng and Shakeel Muhammad and Yui Sudo and William Chen and Jinchuan Tian and Chyi-Jiunn Lin and Shinji Watanabe},
  booktitle={Proceedings of the Annual Conference of the International Speech Communication Association (INTERSPEECH)},
  year={2025},
}
```

#### OWSM-CTC

```BibTex
@inproceedings{owsm-ctc,
    title = "{OWSM}-{CTC}: An Open Encoder-Only Speech Foundation Model for Speech Recognition, Translation, and Language Identification",
    author = "Peng, Yifan  and
      Sudo, Yui  and
      Shakeel, Muhammad  and
      Watanabe, Shinji",
    booktitle = "Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)",
    year = "2024",
    month= {8},
    url = "https://aclanthology.org/2024.acl-long.549",
}
```

#### OWSM v3.1 and v3.2

```BibTex
@inproceedings{owsm-v32,
  title={On the Effects of Heterogeneous Data Sources on Speech-to-Text Foundation Models},
  author={Jinchuan Tian and Yifan Peng and William Chen and Kwanghee Choi and Karen Livescu and Shinji Watanabe},
  booktitle={Proceedings of the Annual Conference of the International Speech Communication Association (INTERSPEECH)},
  year={2024},
  month={9},
  pdf="https://arxiv.org/pdf/2406.09282"
}
@inproceedings{owsm-v31,
  title={{OWSM v3.1: Better and Faster Open Whisper-Style Speech Models based on E-Branchformer}},
  author={Yifan Peng and Jinchuan Tian and William Chen and Siddhant Arora and Brian Yan and Yui Sudo and Muhammad Shakeel and Kwanghee Choi and Jiatong Shi and Xuankai Chang and Jee-weon Jung and Shinji Watanabe},
  booktitle={Proceedings of the Annual Conference of the International Speech Communication Association (INTERSPEECH)},
  year={2024},
  month={9},
  pdf="https://arxiv.org/pdf/2401.16658",
}
```

#### Initial OWSM (v1, v2, v3)

```BibTex
@inproceedings{owsm,
  title={Reproducing Whisper-Style Training Using An Open-Source Toolkit And Publicly Available Data},
  author={Yifan Peng and Jinchuan Tian and Brian Yan and Dan Berrebbi and Xuankai Chang and Xinjian Li and Jiatong Shi and Siddhant Arora and William Chen and Roshan Sharma and Wangyou Zhang and Yui Sudo and Muhammad Shakeel and Jee-weon Jung and Soumi Maiti and Shinji Watanabe},
  booktitle={Proceedings of the IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  year={2023},
  month={12},
  pdf="https://arxiv.org/pdf/2309.13876",
}
```
