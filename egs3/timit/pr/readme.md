# TIMIT phone recognition recipe

Scores [PhoneticXeus](https://huggingface.co/changelinglab/PhoneticXeus), a
multilingual phone recognizer that transcribes speech into IPA, on TIMIT.

## Requirements

TIMIT is licensed by the LDC as [LDC93S1](https://catalog.ldc.upenn.edu/LDC93S1)
and cannot be downloaded automatically; `create_dataset` reports where to put it.
The corpus is only ever read, so a read-only mount is fine. The 2.3 GB checkpoint
is fetched from the Hub on first use.

```bash
pip install -e ".[pr]"
export TIMIT=/path/to/TIMIT          # the directory holding TRAIN/ and TEST/
export TIMIT_OUTPUT=/path/to/scratch # manifest location; defaults to data/
```

## Test set

All **6300** utterances -- both the TRAIN and TEST trees -- form a single
evaluation set, so these scores are comparable to the `PR-tmt` results on the
PRiSM benchmark. This is **not** the 1680-utterance TIMIT test split.

TIMIT's 61-symbol phone labels are converted to IPA with the table in
`dataset/config.yaml`, reproducing the published `text.good` references for all
6300 utterances exactly.

## Quick start

```bash
python run.py --stages create_dataset --training_config conf/training.yaml

python run.py --stages infer --inference_config conf/inference.yaml

python run.py --stages measure \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

## Results

| TIMIT (6300 utterances) | PER | PFER |
|---|---|---|
| this recipe | 42.5 | 13.3 |
| reported in the paper | -- | 13.3 |

The paper reports PFER only.

## Citation

```bibtex
@misc{pxeus26,
      title={An Empirical Recipe for Universal Phone Recognition},
      author={Shikhar Bharadwaj and Chin-Jou Li and Kwanghee Choi and Eunjung Yeo and William Chen and Shinji Watanabe and David R. Mortensen},
      year={2026},
      eprint={2603.29042},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.29042},
}

@inproceedings{bharadwaj-etal-2026-prism,
    title = "{PR}i{SM}: Benchmarking Phone Realization in Speech Models",
    author = "Bharadwaj, Shikhar and Li, Chin-Jou and Kim, Yoonjae and
      Choi, Kwanghee and Yeo, Eunjung and Shim, Ryan Soh-Eun and Zhou, Hanyu and
      Boldt, Brendon and Rosero, Karen and Chang, Kalvin and Agrawal, Darsh and
      Xu, Keer and Yang, Chao-Han Huck and Zhu, Jian and Watanabe, Shinji and
      Mortensen, David R.",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.825/",
    doi = "10.18653/v1/2026.acl-long.825",
    pages = "18093--18112"
}
```
