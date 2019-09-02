# BPE 8k
### RNN baseline (en->de)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train.en-de.de_tc_tc_pytorch_train_bpe8000/decode_tst-COMMON.de_decode|25.24|59.0|31.2|18.8|11.7|1.000|1.000|51449|51459|
|exp/train.en-de.de_tc_tc_pytorch_train_bpe8000/decode_tst-HE.de_decode|23.48|55.1|28.8|17.5|10.9|1.000|1.030|12699|12327|

### RNN baseline (en->pt)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train.en-pt.pt_tc_tc_pytorch_train_bpe8000/decode_tst-COMMON.pt_decode|32.36|62.8|38.8|25.9|17.4|1.000|1.007|50235|49886|
|exp/train.en-pt.pt_tc_tc_pytorch_train_bpe8000/decode_tst-HE.pt_decode|32.64|64.0|40.0|26.6|18.1|0.980|0.980|12612|12863|
