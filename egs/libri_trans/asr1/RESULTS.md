# RNN (character unit)
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc.rm_pytorch_train_rnn_char/decode_dev.en_decode_rnn_char|1071|18651|91.3|7.8|0.8|1.0|**9.6**|62.7|
|exp/train_sp.en_lc.rm_pytorch_train_rnn_char/decode_test.en_decode_rnn_char|2048|36336|90.7|8.3|0.9|1.2|**10.4**|62.1|

# RNN (character unit v2)
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc.rm_pytorch_train_bpe38/decode_dev.en_decode_rnn_char|1071|18651|92.4|6.9|0.7|1.0|**8.6**|58.3|
|exp/train_sp.en_lc.rm_pytorch_train_bpe38/decode_test.en_decode_rnn_char|2048|36336|92.2|7.0|0.8|1.1|**8.9**|56.5|

# RNN (BPE1k)
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc.rm_pytorch_train_rnn_bpe_bpe1000/decode_dev.en_decode_rnn_char|1071|18651|93.4|6.0|0.6|1.1|**7.7**|54.3|
|exp/train_sp.en_lc.rm_pytorch_train_rnn_bpe_bpe1000/decode_test.en_decode_rnn_char|2048|36336|93.1|6.1|0.8|1.0|**7.9**|53.1|


# Transformer (character unit v2)
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/ttrain_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe38/decode_dev.en_decode_pytorch_transformer|1071|18651|93.1|6.2|0.6|1.1|**8.0**|52.3|
|exp/ttrain_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe38/decode_test.en_decode_pytorch_transformer|2048|36336|92.7|6.7|0.7|1.0|**8.4**|53.9|

# Transformer (BPE1k)
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000/decode_dev.en_decode_rnn_char|1071|18651|93.9|5.5|0.7|1.0|**7.1**|52.5|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000/decode_test.en_decode_rnn_char|2048|36336|93.8|5.5|0.7|0.9|**7.2**|50.0|
