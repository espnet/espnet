- Environments
  - date `Thursday June 4, 2020`
  - operating system `Ubuntu 18.04.4 LTS (GNU/Linux 5.3.0-1020-gcp x86_64)`
  - python version: `3.6.9 [GC 7.5.0]`
  - espnet version: `espnet 0.7.0`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.0.1`


- Results(`batch-size 16` `transformer-lr 20`using SpecAug)
  ### CER

  |dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
  |---|---|---|---|---|---|---|---|---|
  |decode_test_decode_lm_word65000|100|5767|89.6|5.0|5.4|2.3|12.7|72.0
  ### WER(removed sentences including abbreviations because there is no abbreviations in timit corpus)

  |dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
  |---|---|---|---|---|---|---|---|---|
  |decode_test_decode_lm_word65000|89|886|81.3|15.9|2.8|3.0|21.8|68.5 |
- Results(`batch-size 12` `transformer-lr 20` )

  ### CER

  |dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
  |---|---|---|---|---|---|---|---|---|
  |decode_test_decode_lm_word65000|100|5767|81.3|8.1|10.5|4.0|22.6|90.0 
  |decode_test_decode_lm_word8000|100|5767|81.3|8.1|10.5|4.0|22.6|90.0 
  ### WER(lm_word8000 means language model trained using timit corpus)

  |dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
  |---|---|---|---|---|---|---|---|---|
  |decode_test_decode_lm_word65000|100|1023|63.1|30.4|6.5|5.2|42.0|90.0
  |decode_test_decode_lm_word8000|100|1023|62.8|30.7|6.5|5.0|42.2|88.0 
  
