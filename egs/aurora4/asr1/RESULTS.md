# Transformer result(default transformer with n_average = 10, epoch = 100)
  
  - Environments (obtained by `$ get_sys_info.sh`)
    - date: `Thu Jun 13 01:29:55 EDT 2019`
    - system information: `Linux a09 4.9.0-4-amd64 #1 SMP Debian 4.9.65-3+deb9u1 (2017-12-23) x86_64 GNU/Linux`
    - python version: `Python 3.6.8 :: Anaconda, Inc.
    - espnet version: `espnet 0.3.1`
    - chainer version: `chainer 5.0.0`
    - pytorch version: `pytorch 1.0.0`
    - Git hash: `04b9e1a71b3272cdf6502e40679ab17f3f3f22f6`

```
exp/train_mix_pytorch_transformer_100epoch/decode_dev_0330_decode_lm_word65000/result.wrd_A.txt:|  SPKR     |  # Snt    # Wrd   |  Corr        Sub       Del       Ins       Err     S.Err   |
exp/train_mix_pytorch_transformer_100epoch/decode_dev_0330_decode_lm_word65000/result.wrd_A.txt:|  Sum/Avg  |   330      5467   |  96.6        3.1       0.3       0.3       3.7      33.0   |
exp/train_mix_pytorch_transformer_100epoch/decode_dev_0330_decode_lm_word65000/result.wrd_B.txt:|  SPKR     |  # Snt     # Wrd   |  Corr       Sub       Del       Ins       Err     S.Err   |
exp/train_mix_pytorch_transformer_100epoch/decode_dev_0330_decode_lm_word65000/result.wrd_B.txt:|  Sum/Avg  |  1980      32802   |  92.4       6.4       1.2       1.0       8.6      53.3   |
exp/train_mix_pytorch_transformer_100epoch/decode_dev_0330_decode_lm_word65000/result.wrd_C.txt:|  SPKR     |  # Snt    # Wrd   |  Corr        Sub       Del       Ins       Err     S.Err   |
exp/train_mix_pytorch_transformer_100epoch/decode_dev_0330_decode_lm_word65000/result.wrd_C.txt:|  Sum/Avg  |   330      5467   |  93.9        5.3       0.7       0.7       6.7      43.6   |
exp/train_mix_pytorch_transformer_100epoch/decode_dev_0330_decode_lm_word65000/result.wrd_D.txt:|  SPKR     |  # Snt     # Wrd   |  Corr       Sub       Del       Ins       Err     S.Err   |
exp/train_mix_pytorch_transformer_100epoch/decode_dev_0330_decode_lm_word65000/result.wrd_D.txt:|  Sum/Avg  |  1980      32802   |  85.7      12.2       2.1       2.0      16.3      65.2   |
```

# Best result for RNN based: mix SI284 + shallow_wide network + wordlm

```
exp/train_mix_pytorch_mix_shallow_wide/decode_dev_0330_decode_wordlm/result.wrd_A.txt:|  SPKR    |  # Snt   # Wrd  |  Corr      Sub      Del      Ins      Err    S.Err  |
exp/train_mix_pytorch_mix_shallow_wide/decode_dev_0330_decode_wordlm/result.wrd_A.txt:|  Sum/Avg |   330     5467  |  96.3      3.2      0.5      0.4      4.1     30.0  |
exp/train_mix_pytorch_mix_shallow_wide/decode_dev_0330_decode_wordlm/result.wrd_B.txt:|  SPKR    |  # Snt    # Wrd  |  Corr     Sub      Del      Ins      Err    S.Err  |
exp/train_mix_pytorch_mix_shallow_wide/decode_dev_0330_decode_wordlm/result.wrd_B.txt:|  Sum/Avg |  1980     32802  |  91.8     6.8      1.4      0.9      9.0     51.5  |
exp/train_mix_pytorch_mix_shallow_wide/decode_dev_0330_decode_wordlm/result.wrd_C.txt:|  SPKR    |  # Snt   # Wrd  |  Corr      Sub      Del      Ins      Err    S.Err  |
exp/train_mix_pytorch_mix_shallow_wide/decode_dev_0330_decode_wordlm/result.wrd_C.txt:|  Sum/Avg |   330     5467  |  93.5      5.7      0.8      0.8      7.3     43.6  |
exp/train_mix_pytorch_mix_shallow_wide/decode_dev_0330_decode_wordlm/result.wrd_D.txt:|  SPKR    |  # Snt    # Wrd  |  Corr     Sub      Del      Ins      Err    S.Err  |
exp/train_mix_pytorch_mix_shallow_wide/decode_dev_0330_decode_wordlm/result.wrd_D.txt:|  Sum/Avg |  1980     32802  |  83.6    14.3      2.0      2.2     18.5     65.0  |
```

# First result
```
exp/train_si84_multi_pytorch_first/decode_dev_0330_decode_nolm/result.wrd_A.txt: | SPKR    |  # Snt  # Wrd  | Corr      Sub     Del     Ins      Err   S.Err  |
exp/train_si84_multi_pytorch_first/decode_dev_0330_decode_nolm/result.wrd_A.txt: | Sum/Avg |   330    5467  | 57.3     39.5     3.2     7.2     49.9    99.7  |
exp/train_si84_multi_pytorch_first/decode_dev_0330_decode_nolm/result.wrd_B.txt: | SPKR    | # Snt    # Wrd  | Corr     Sub     Del      Ins     Err   S.Err  |
exp/train_si84_multi_pytorch_first/decode_dev_0330_decode_nolm/result.wrd_B.txt: | Sum/Avg | 1980     32802  | 49.1    46.7     4.2      8.2    59.2    99.9  |
exp/train_si84_multi_pytorch_first/decode_dev_0330_decode_nolm/result.wrd_C.txt: | SPKR    |  # Snt  # Wrd  | Corr      Sub     Del     Ins      Err   S.Err  |
exp/train_si84_multi_pytorch_first/decode_dev_0330_decode_nolm/result.wrd_C.txt: | Sum/Avg |   330    5467  | 49.0     47.7     3.3     8.8     59.7   100.0  |
exp/train_si84_multi_pytorch_first/decode_dev_0330_decode_nolm/result.wrd_D.txt: | SPKR    | # Snt    # Wrd  | Corr     Sub     Del      Ins     Err   S.Err  |
exp/train_si84_multi_pytorch_first/decode_dev_0330_decode_nolm/result.wrd_D.txt: | Sum/Avg | 1980     32802  | 39.0    56.7     4.3      9.3    70.3    99.9  |
```

# Shallow_wide network without wordlm
```
exp/train_si84_multi_pytorch_wide_shallow_network/decode_dev_0330_decode_lm/result.wrd_A.txt:|  SPKR     |  # Snt    # Wrd  |  Corr       Sub       Del       Ins      Err     S.Err   |
exp/train_si84_multi_pytorch_wide_shallow_network/decode_dev_0330_decode_lm/result.wrd_A.txt:|  Sum/Avg  |   330      5467  |  74.5      22.9       2.6       3.7     29.3      93.6   |
exp/train_si84_multi_pytorch_wide_shallow_network/decode_dev_0330_decode_lm/result.wrd_B.txt:|  SPKR     |  # Snt    # Wrd   |  Corr       Sub      Del       Ins       Err    S.Err   |
exp/train_si84_multi_pytorch_wide_shallow_network/decode_dev_0330_decode_lm/result.wrd_B.txt:|  Sum/Avg  |  1980     32802   |  65.5      31.0      3.5       4.6      39.1     97.0   |
exp/train_si84_multi_pytorch_wide_shallow_network/decode_dev_0330_decode_lm/result.wrd_C.txt:|  SPKR     |  # Snt    # Wrd  |  Corr       Sub       Del       Ins      Err     S.Err   |
exp/train_si84_multi_pytorch_wide_shallow_network/decode_dev_0330_decode_lm/result.wrd_C.txt:|  Sum/Avg  |   330      5467  |  66.6      30.6       2.8       4.4     37.7      97.3   |
exp/train_si84_multi_pytorch_wide_shallow_network/decode_dev_0330_decode_lm/result.wrd_D.txt:|  SPKR     |  # Snt    # Wrd   |  Corr       Sub      Del       Ins       Err    S.Err   |
exp/train_si84_multi_pytorch_wide_shallow_network/decode_dev_0330_decode_lm/result.wrd_D.txt:|  Sum/Avg  |  1980     32802   |  53.5      42.5      4.1       6.3      52.9     98.8   |
```

# Shallow_wide network with wordlm
```
exp/train_si84_multi_pytorch_wide_shallow_network/decode_dev_0330_decode_wordlm/result.wrd_A.txt:|  SPKR     |  # Snt     # Wrd   |  Corr       Sub       Del        Ins       Err     S.Err   |
exp/train_si84_multi_pytorch_wide_shallow_network/decode_dev_0330_decode_wordlm/result.wrd_A.txt:|  Sum/Avg  |   330       5467   |  91.2       7.4       1.4        1.1      10.0      57.9   |
exp/train_si84_multi_pytorch_wide_shallow_network/decode_dev_0330_decode_wordlm/result.wrd_B.txt:|  SPKR     |  # Snt     # Wrd   |  Corr        Sub       Del       Ins       Err     S.Err   |
exp/train_si84_multi_pytorch_wide_shallow_network/decode_dev_0330_decode_wordlm/result.wrd_B.txt:|  Sum/Avg  |  1980      32802   |  86.0       11.8       2.2       1.9      15.9      69.5   |
exp/train_si84_multi_pytorch_wide_shallow_network/decode_dev_0330_decode_wordlm/result.wrd_C.txt:|  SPKR     |  # Snt     # Wrd   |  Corr       Sub       Del        Ins       Err     S.Err   |
exp/train_si84_multi_pytorch_wide_shallow_network/decode_dev_0330_decode_wordlm/result.wrd_C.txt:|  Sum/Avg  |   330       5467   |  87.0      11.5       1.5        2.2      15.2      65.2   |
exp/train_si84_multi_pytorch_wide_shallow_network/decode_dev_0330_decode_wordlm/result.wrd_D.txt:|  SPKR     |  # Snt     # Wrd   |  Corr        Sub       Del       Ins       Err     S.Err   |
exp/train_si84_multi_pytorch_wide_shallow_network/decode_dev_0330_decode_wordlm/result.wrd_D.txt:|  Sum/Avg  |  1980      32802   |  75.6       21.5       2.9       3.8      28.2      79.8   |
```
