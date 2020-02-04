# Summary (BLEU)
|model               |De   |Pt   |Fr   |Es   |Ro   |Ru   |Nl   |It   |
|--------------------|-----|-----|-----|-----|-----|-----|-----|-----|
|Transformer + ASR-PT [[Di Gangi et al.]](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/3045.pdf)|17.30|20.10|26.90|20.80|16.50|10.50|18.80|16.80|
|Pipeline [[Di Gangi et al.]](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/3045.pdf)|18.50|21.50|27.90|22.50|16.80|11.10|22.20|18.90|
|RNN                 |15.97|N/A  |N/A  |N/A  |N/A  |N/A  |N/A  |N/A  |
|Transformer         |16.40|N/A  |N/A  |N/A  |N/A  |N/A  |N/A  |N/A  |
|Transformer + ASR-PT|[21.77](https://drive.google.com/open?id=18zlGTLcwgr0PF1b6eZIANwJSMyOqqEjd)|[26.84](https://drive.google.com/open?id=1-vdZDN0YimYcrx3yearGgp5ismcVcIYr)|[31.56](https://drive.google.com/open?id=1GvUdGbBP2w7vPxIAD1i1r22oa_qABJev)|[26.41](https://drive.google.com/open?id=1ecd7FwWzc0p2HBHIvBUn7ImlL3bdjfMt)|[20.53](https://drive.google.com/open?id=1PZ-oqbrgBttUp8SyZy38zS_B07UW8GTG)|[14.31](https://drive.google.com/open?id=1IFG8TT_Shx3eJld3d7pNI5GrIHhTaIly)|[25.22](https://drive.google.com/open?id=19RbO7xkXBgGFXFXskqjpMrTr467ltuh2)|[21.46](https://drive.google.com/open?id=19Rf6DgibGJ8WTpVZcslmLPxTLl56FB-s)|
|Transformer + ASR-PT + MT-PT |[22.06](https://drive.google.com/open?id=1jWmlGq5pzaKJsZ7SQmDXGIL3UEcSqWwp)|N/A  |[31.65](https://drive.google.com/open?id=1wFIAqxoBUioTKTLRLv29KzvphkUm3qdo)|[27.49](https://drive.google.com/open?id=1wFIAqxoBUioTKTLRLv29KzvphkUm3qdo)|N/A  |[15.64](https://drive.google.com/open?id=1wJ537O6iQPdrcLypx7Aa5c-a8Yz7uUH9)|N/A  |N/A  |


### RNN (tc, BPE8k)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en-de.de_tc_pytorch_train_rnn_bpe8000/decode_tst-COMMON.en-de.de_decode|16.13|48.5|21.1|11.0|6.0|1.000|1.001|51532|51459|
|exp/train_sp.en-de.de_tc_pytorch_train_rnn_bpe8000/decode_tst-HE.en-de.de_decode|15.17|46.1|19.6|10.2|5.7|1.000|1.029|12682|12327|


### Transformer (tc, BPE8k)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en-de.de_tc_pytorch_train_pytorch_transformer_bpe8000/decode_tst-COMMON.en-de.de_decode_pytorch_transformer.en-de|16.98|49.8|22.2|12.0|6.8|0.980|0.980|50439|51459|
|exp/train_sp.en-de.de_tc_pytorch_train_pytorch_transformer_bpe8000/decode_tst-HE.en-de.de_decode_pytorch_transformer.en-de|16.15|47.4|20.8|11.2|6.3|0.998|0.998|12301|12327|


### Transformer (tc, BPE8k) + ASR pre-training
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en-de.de_tc_pytorch_train_pytorch_transformer_short_bpe8000_asrtrans/decode_tst-COMMON.en-de.de_decode_pytorch_transformer.en-de|21.77|54.7|27.3|15.8|9.7|0.996|0.996|51260|51459|
|exp/train_sp.en-de.de_tc_pytorch_train_pytorch_transformer_short_bpe8000_asrtrans/decode_tst-HE.en-de.de_decode_pytorch_transformer.en-de|19.91|52.0|25.1|14.3|8.5|1.000|1.011|12457|12327|


### Transformer (tc, BPE8k) + ASR pre-training + MT-pre-training
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en-de.de_tc_pytorch_train_pytorch_transformer_short_bpe8000_asrtrans_mttrans/decode_tst-COMMON.en-de.de_decode_pytorch_transformer.en-de|22.06|55.2|27.7|16.2|10.0|0.990|0.991|50972|51459|
|exp/train_sp.en-de.de_tc_pytorch_train_pytorch_transformer_short_bpe8000_asrtrans_mttrans/decode_tst-HE.en-de.de_decode_pytorch_transformer.en-de|20.94|52.8|25.9|15.2|9.3|1.000|1.007|12411|12327|
