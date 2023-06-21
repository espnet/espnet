# NOTE: apostrophe is included both in hyp and ref

# Summary (4-gram BLEU)

| model                                                         | fisher_dev | fisher_dev2 | fisher_test | callhome_devtest | callhome_evltest |
| ------------------------------------------------------------- | ---------- | ----------- | ----------- | ---------------- | ---------------- |
| RNN (char) [[Weiss et al.]](https://arxiv.org/abs/1703.08581) | 48.3       | 49.1        | 48.7        | 16.8             | 17.4             |
| Transformer (BPE1k(500ES,500EN)) + ASR-PT + SpecAugment       | 48.4       | 49.5        | 48.6        | 19.7             | 19.6             |
| Conformer (BPE1k(500ES,500EN)) + ASR-PT + SpecAugment         | **51.8**   | **52.3**    | **50.5**    | **22.3**         | **21.7**         |

# Summary (4-gram BLEU, no callhome training)

| model                                                         | fisher_dev | fisher_dev2 | fisher_test | callhome_devtest | callhome_evltest |
| ------------------------------------------------------------- | ---------- | ----------- | ----------- | ---------------- | ---------------- |
| Transformer (BPE1k(500ES,500EN)) + SpecAugment                | 44.7       | 45.6        | 45.1        | 17.3             | 16.8             |