# IPAPack++ Recipe
This is a S2T recipe for [IPAPack++](https://huggingface.co/anyspeech), including ASR, phone recognition, G2P and P2G as pretraining task.

## Guidelines
1. Download IPAPack++: `local/data.sh --stage 0 --stop_stage 0`
2. Data prep: stage 1-4, can be done without GPU. The extracted `dump/raw` is around 1T.
3. Train BPE: stage 5
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