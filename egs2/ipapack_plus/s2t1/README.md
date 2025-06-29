# IPAPack++ Recipe
This is a S2T recipe for [IPAPack++](https://huggingface.co/anyspeech), including ASR, phone recognition, G2P and P2G as pretraining task.

## Guidelines
1. Download IPAPack++: `local/data.sh --stage 0 --stop_stage 0`
    - Raw data will be located at `downloads/`
    - `transcripts.csv` contains orthography, phones, and path to audio file
2. Data prep: stage 1-4, can be done without GPU. The extracted `dump/raw` is around 1T.
    - `local/data_prep.py` filters out unused entries and normalizes phones with the provided mapping `local/ipa_mapping.json`, such as removing diacritics
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


```
@article{zhu2025zipa,
  title={ZIPA: A family of efficient models for multilingual phone recognition},
  author={Zhu, Jian and Samir, Farhan and Chodroff, Eleanor and Mortensen, David R},
  journal={arXiv preprint arXiv:2505.23170},
  year={2025}
}
```

## Results

TBD
