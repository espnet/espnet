### RNN (BPE8k unit)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en-de.de_tc_pytorch_train_rnn_bpe8000/decode_tst-COMMON.en-de.de_decode|16.13|48.5|21.1|11.0|6.0|1.000|1.001|51532|51459|
|exp/train_sp.en-de.de_tc_pytorch_train_rnn_bpe8000/decode_tst-HE.en-de.de_decode|15.17|46.1|19.6|10.2|5.7|1.000|1.029|12682|12327|


### Transformer (BPE8k unit)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en-de.de_tc_pytorch_train_pytorch_transformer_bpe8000/decode_tst-COMMON.en-de.de_decode|16.98|49.8|22.2|12.0|6.8|0.980|0.980|50439|51459|
|exp/train_sp.en-de.de_tc_pytorch_train_pytorch_transformer_bpe8000/decode_tst-HE.en-de.de_decode|16.15|47.4|20.8|11.2|6.3|0.998|0.998|12301|12327|
