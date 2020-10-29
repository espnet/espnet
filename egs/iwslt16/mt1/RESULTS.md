# IWSLT 2016 Machine Translation

## Preprocesssing

- Moses `tokenizer`
- Moses `truecaser`
- Moses `clean-corpus-n`
- BPE splitting using [subword-nmt](https://github.com/rsennrich/subword-nmt)
    - number of merge operations: 16000

## Model Configuration

- Model: Transformer
    - number of layers: 6
    - d_model (adim): 256
    - d_ff (eunits/dunits): 2048
    - Gradient Clipping: 5
    - Dropout Rate: 0.1
    - Warmup steps: 8000

## Result

These results will be updated soon.


|           |         | En-->De |         |         | De-->en |         |
|-----------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Framework | tst2012 | tst2013 | tst2014 | tst2012 | tst2013 | tst2014 |
|  Fairseq  |   27.73 |   29.45 |   25.14 |   32.25 |   34.23 |   29.49 |
|   ESPNet  |   26.92 |   28.88 |   24.70 |   32.19 |   33.46 |   29.22 |


## Pretrained Model

[Link](https://drive.google.com/open?id=1o150rdEhPubN1i36TK_dWktFkxuL-X19)
