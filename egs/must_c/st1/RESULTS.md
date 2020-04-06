# Summary (4-gram BLEU)
|model               |De   |Pt   |Fr   |Es   |Ro   |Ru   |Nl   |It   |
|--------------------|-----|-----|-----|-----|-----|-----|-----|-----|
|Transformer + ASR-PT [[Di Gangi et al.]](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/3045.pdf)|17.30|20.10|26.90|20.80|16.50|10.50|18.80|16.80|
|Pipeline [[Di Gangi et al.]](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/3045.pdf)|18.50|21.50|27.90|22.50|16.80|11.10|22.20|18.90|
|RNN                                       |15.97|N/A  |N/A  |N/A  |N/A  |N/A  |N/A  |N/A  |
|Transformer                               |16.40|N/A  |N/A  |N/A  |N/A  |N/A  |N/A  |N/A  |
|Transformer + ASR-MTL                     |22.31|N/A  |N/A  |N/A  |N/A  |N/A  |N/A  |N/A  |
|Transformer + ASR-MTL + MT-MTL            |21.77|N/A  |N/A  |N/A  |N/A  |N/A  |N/A  |N/A  |
|Transformer + ASR-PT                      |21.77|26.84|31.56|26.41|20.53|14.31|25.22|21.46|
|Transformer + ASR-PT + MT-PT              |22.33|27.26|31.54|27.84|20.91|15.32|26.86|22.81|
|Transformer + ASR-PT + MT-PT + SpecAugment|[22.91](https://drive.google.com/open?id=1KYduzn-500vbo1TO2WdxvE4za71BvrRA)|[28.01](https://drive.google.com/open?id=1PWDXlSpo8J-fj6S3FWxBWNqpsfPShlyV)|[32.76](https://drive.google.com/open?id=1mnOIjwu79Iw5B1eLP2boa9D6HUyi9_e2)|[27.96](https://drive.google.com/open?id=1kEaFjwd18OxJ5Xf0ylEWlWxwHF-fPi4N)|[21.90](https://drive.google.com/open?id=1j5izknFXzcAU-IOCSz9WALl6aEGSonUM)|[15.75](https://drive.google.com/open?id=1D0SB1t1wU4_lP7FQrD2ucZorFbWCv0jx)|[27.43](https://drive.google.com/open?id=1eF_LVCpfjTH5I97qcXm8AdShsTtfQtLg)|[23.75](https://drive.google.com/open?id=1Wjs7yEcNNUrc94pPzm-phVXrmrZZ2wgs)|


# Transformer results
### train_sp.en-de.de_tc_pytorch_train_pytorch_transformer_short_long_bpe8000_specaug_asrtrans_mttrans
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en-de.de_tc_pytorch_train_pytorch_transformer_short_long_bpe8000_specaug_asrtrans_mttrans/decode_tst-COMMON.en-de.de_decode_pytorch_transformer.en-de|**22.91**|56.0|28.6|17.0|10.5|0.991|0.992|51023|51459|
|exp/train_sp.en-de.de_tc_pytorch_train_pytorch_transformer_short_long_bpe8000_specaug_asrtrans_mttrans/decode_tst-HE.en-de.de_decode_pytorch_transformer.en-de|**22.27**|53.1|27.1|16.4|10.4|1.000|1.014|12501|12327|
- NOTE: longer version of "short" for SpecAugment: 30ep->50ep

### train_sp.en-de.de_tc_pytorch_train_pytorch_transformer_short_bpe8000_asrtrans_mttrans
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en-de.de_tc_pytorch_train_pytorch_transformer_short_bpe8000_asrtrans_mttrans/decode_tst-COMMON.en-de.de_decode_pytorch_transformer.en-de|**22.33**|55.5|28.1|16.5|10.2|0.986|0.986|50721|51459|
|exp/train_sp.en-de.de_tc_pytorch_train_pytorch_transformer_short_bpe8000_asrtrans_mttrans/decode_tst-HE.en-de.de_decode_pytorch_transformer.en-de|**21.59**|52.9|26.4|15.9|9.8|1.000|1.004|12380|12327|
- NOTE: shorten the total number epochs when pre-training the model: 100ep->30ep

### train_sp.en-de.de_tc_pytorch_train_pytorch_transformer_short_bpe8000_asrtrans
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en-de.de_tc_pytorch_train_pytorch_transformer_short_bpe8000_asrtrans/decode_tst-COMMON.en-de.de_decode_pytorch_transformer.en-de|**21.77**|54.7|27.3|15.8|9.7|0.996|0.996|51260|51459|
|exp/train_sp.en-de.de_tc_pytorch_train_pytorch_transformer_short_bpe8000_asrtrans/decode_tst-HE.en-de.de_decode_pytorch_transformer.en-de|**19.91**|52.0|25.1|14.3|8.5|1.000|1.011|12457|12327|
- NOTE: shorten the total number epochs when pre-training the model: 100ep->30ep


# RNN results
### train_sp.en-de.de_tc_pytorch_train_rnn_bpe8000
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en-de.de_tc_pytorch_train_rnn_bpe8000/decode_tst-COMMON.en-de.de_decode|**16.13**|48.5|21.1|11.0|6.0|1.000|1.001|51532|51459|
|exp/train_sp.en-de.de_tc_pytorch_train_rnn_bpe8000/decode_tst-HE.en-de.de_decode|**15.17**|46.1|19.6|10.2|5.7|1.000|1.029|12682|12327|
