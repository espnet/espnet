# Transformer (large model + specaug) (4 GPUs)
- KsponSpeech: Korean Spontaneous Speech Corpus for Automatic Speech Recognition
    - Database: https://aihub.or.kr/aidata/105
    - Paper: https://www.mdpi.com/2076-3417/10/19/6936
    - This corpus contains 969 h of general open-domain dialog utterances, spoken by 2000 native Korean speakers.

## Environments
- date: `Mon Oct  5 11:11:27 KST 2020`
- python version: `3.8.3 (default, May 19 2020, 18:47:26)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.1`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.4.0`

- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: ([pretrained model](https://drive.google.com/file/d/1aoAxAjZzqCoP4MheZzUE2CHTHNQJdcD_/view?usp=sharing))
    - training config file: `conf/tuning/train_pytorch_transformer_large_ngpu4.yaml`
    - decoding config file: `conf/tuning/decode_pytorch_transformer_large.yaml`
    - cmvn file: `data/train/cmvn.ark`
    - e2e file: `exp/train_pytorch_train_specaug/results/model.val5.avg.best`
    - e2e JSON file: `exp/train_pytorch_train_specaug/results/model.json`
    - dict file: `data/lang_char`

## CER
```
exp/train_pytorch_train_specaug/decode_eval_clean_model.val5.avg.best_decode_lm/result.txt
| SPKR                | # Snt   # Wrd  | Corr     Sub     Del      Ins     Err   S.Err  |
| Sum/Avg             | 3000    68457  | 94.2     3.5     2.3      1.9     7.7    59.6  |
exp/train_pytorch_train_specaug/decode_eval_other_model.val5.avg.best_decode_lm/result.txt
| SPKR                | # Snt   # Wrd  | Corr     Sub     Del      Ins     Err   S.Err  |
| Sum/Avg             | 3000    95616  | 93.8     3.8     2.4      2.2     8.4    71.3  |
```
## WER
```
exp/train_pytorch_train_specaug/decode_eval_clean_model.val5.avg.best_decode_lm/result.wrd.txt
|  SPKR                | # Snt   # Wrd  |  Corr     Sub      Del      Ins     Err    S.Err  |
|  Sum/Avg             | 3000    20401  |  82.4    14.5      3.1      3.8    21.4     59.5  |
exp/train_pytorch_train_specaug/decode_eval_other_model.val5.avg.best_decode_lm/result.wrd.txt
|  SPKR                | # Snt   # Wrd  |  Corr     Sub      Del      Ins     Err    S.Err  |
|  Sum/Avg             | 3000    26621  |  79.7    17.1      3.1      5.0    25.3     71.3  |
```
## sWER (space-normalized WER)
- This metric was measured from space-normalized texts, which was performed only on the hypothesis text, based on spaces in the reference text. In Korean, space rules are flexible; inconsistent spacing is frequently seen in spontaneous speech transcriptions, like in KsponSpeech. However, this causes a problem in the evaluation of speech recognition because correct results are classified as errors due to this spacing variation. Thus, we used sWER, which gives a more valid word error rate by excluding the effects of inconsistent spaces. A more detailed description is given in our paper.
```
exp/train_pytorch_train_specaug/decode_eval_clean_model.val5.avg.best_decode_lm/result.wrd.sp_norm.txt
|  SPKR                 |  # Snt   # Wrd   |  Corr      Sub       Del      Ins       Err    S.Err   |
|  Sum/Avg              |  3000    20401   |  87.8     10.7       1.5      1.4      13.6     50.4   |
exp/train_pytorch_train_specaug/decode_eval_other_model.val5.avg.best_decode_lm/result.wrd.sp_norm.txt
|  SPKR                 |  # Snt   # Wrd   |  Corr      Sub       Del      Ins       Err    S.Err   |
|  Sum/Avg              |  3000    26621   |  86.3     12.3       1.4      1.5      15.1     61.1   |
```

