# First result with default Transformer setting
## bpemode=unigram
### CER
```
exp/valid_train_pytorch_train_unigram/decode_valid_dev_decode_lm/result.txt
| SPKR                        | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                     | 4076  100570| 98.3    1.0    0.7    0.3    2.0   11.8 |
exp/valid_train_pytorch_train_unigram/decode_valid_test_decode_lm/result.txt
| SPKR                        | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                     | 3995  99017 | 98.3    1.0    0.7    0.3    1.9   10.8 |
```

### WER
```
exp/valid_train_pytorch_train_unigram/decode_valid_dev_decode_lm/result.wrd.txt
| SPKR                        | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                     | 4076  38374 | 97.8    1.8    0.4    0.4    2.5   11.8 |
exp/valid_train_pytorch_train_unigram/decode_valid_test_decode_lm/result.wrd.txt
| SPKR                        | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                     | 3995  37837 | 97.9    1.7    0.4    0.3    2.4   10.8 |
```
