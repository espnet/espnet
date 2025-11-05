# 🐁POWSM

POWSM is the first phonetic foundation model that can perform four phone-related tasks:
Phone Recognition (PR), Automatic Speech Recognition (ASR), audio-guided grapheme-to-phoneme conversion (G2P), and audio-guided phoneme-to-grapheme
conversion (P2G).

Based on [Open Whisper-style Speech Model (OWSM)](https://www.wavlab.org/activities/2024/owsm/) and trained with [IPAPack++](https://huggingface.co/anyspeech), POWSM outperforms or matches specialized PR models of similar size (Wav2Vec2Phoneme and ZIPA) while jointly supporting G2P, P2G, and ASR.

The model checkpoint is available on [HuggingFace](https://huggingface.co/espnet/powsm)!


## Results

Please refer to our paper for comprehensive evaluations, while below are selected results on phone recognition:

### Unseen languages

| Model             | Param. | DoReCo  | VoxAngeles | Tusom2021 | Avg.  |
|------------------|--------|---------|------------|-----------|-------|
| Allosaurus       | 11M    | 24.71   | 30.84      | 42.02     | 32.52 |
| Allophant        | 300M   | ---     | ---        | ---       | ---   |
| Wav2Vec2Phoneme  | 300M   | 17.25   | **13.88**  | 31.92     | 21.02 |
| MultIPA          | 300M   | 18.28   | 15.23      | 30.53     | 21.35 |
| ZIPA-CR-NS-Large | 300M   | **16.82** | 17.14    | 23.08     | 19.01 |
| POWSM            | 350M   | 17.06   | 17.11      | **21.96** | **18.71** |

### Sociophonetic variation of English

| Model             | Param. | Buckeye | DRC-SE | L2-ARC | EpaDB  | SO762  | Avg.  |
|------------------|--------|--------|--------|--------|--------|--------|-------|
| Allosaurus       | 11M    | 15.24  | 25.36  | 13.39  | 19.33  | 21.61  | 18.99 |
| Allophant        | 300M   | 16.05  | 24.13  | 11.91  | 14.38  | 18.28  | 16.95 |
| Wav2Vec2Phoneme  | 300M   | 12.50  | 18.57  | 9.86   | **9.90** | **13.60** | **12.89** |
| MultIPA          | 300M   | 18.69  | 23.31  | 15.52  | 15.64  | 21.34  | 18.90 |
| ZIPA-CR-NS-Large | 300M   | **12.05**  | **17.12** | **9.69** | 14.63 | 18.20 | 14.34 |
| POWSM            | 350M   | 12.63  | 18.33  | 11.32  | 11.86  | 17.84  | 14.40 |


## Guidelines
1. Download IPAPack++: `local/data.sh --stage 0 --stop_stage 0`
    - Raw data will be located at `downloads/`
    - `transcripts.csv` contains orthography, phones, and path to audio file
2. Data prep: stage 1-4, can be done without GPU. The extracted `dump/raw` is around 1T.
    - `local/data_prep.py` normalizes phones into [PanPhon](https://github.com/dmort27/panphon) phone entries through greedy trie search
    - A new transcript `transcripts_normalized.csv` is then generated
    - Wav files are dumped to ark files in `data/format.{i}/`
    - `local/process_ipapack.py` generates OWSM format text files for each task, which looks like: `uttid_task <lang><task><notimestamp> text`
    - `local/subset.py` combines text files and generate other files accordingly. It also has other functions to get subsets in different ways
3. Train BPE: stage 5
    - `data/nlsyms.txt` and `data/bpe_nlsyms.txt` define symbols that should be regarded as a token, such as task tokens and phones
    - Since `train.txt` is large, this step requires quite a lot of GPU memory
4. (stage 6-9 are skipped automatically)
5. Collect stats and train: stage 10-11
6. Eval: stage 12-13, compute intensive, decode each dataset seperately to set language and task token.


### Citations

```BibTex
@article{powsm,
      title={POWSM: A Phonetic Open Whisper-Style Speech Foundation Model},
      author={Chin-Jou Li and Kalvin Chang and Shikhar Bharadwaj and Eunjung Yeo and Kwanghee Choi and Jian Zhu and David Mortensen and Shinji Watanabe},
      year={2025},
      eprint={2510.24992},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.24992},
}

@inproceedings{zhu-etal-2025-zipa,
    title = "{ZIPA}: A family of efficient models for multilingual phone recognition",
    author = "Zhu, Jian  and  Samir, Farhan  and  Chodroff, Eleanor  and  Mortensen, David R.",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.961/",
}
```
