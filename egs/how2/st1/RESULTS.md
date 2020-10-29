# Transformer results
### ensemble (1) + (2) + (3)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train.pt_tc_pytorch_train_pytorch_transformer_short_long_bpe8000_specaug_asrtrans_mttrans/decode_dev5.pt_decode_ensemble3|**48.04**|73.8|54.2|41.8|32.3|0.996|0.996|43907|44062|

### train.pt_tc_pytorch_train_pytorch_transformer_short_long_bpe8000_specaug_asrtrans_mttrans (1)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train.pt_tc_pytorch_train_pytorch_transformer_short_long_bpe8000_specaug_asrtrans_mttrans/decode_dev5.pt_decode|**45.68**|71.6|51.6|39.2|30.0|1.000|1.001|44103|44062|

- Model files (archived to train.pt_tc_pytorch_train_pytorch_transformer_short_long_bpe8000_specaug_asrtrans_mttrans.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1VFCOPngnBqVl-RAJxr0HDzeLHyk3yOfZ
  - training config file: `conf/train_pytorch_transformer_short_long.yaml`
  - decoding config file: `conf/decode.yaml`
  - cmvn file: `data/train.pt/cmvn.ark`
  - e2e file: `exp/train.pt_tc_pytorch_train_pytorch_transformer_short_long_bpe8000_specaug_asrtrans_mttrans/results/model.val5.avg.best`
  - e2e JSON file: `exp/train.pt_tc_pytorch_train_pytorch_transformer_short_long_bpe8000_specaug_asrtrans_mttrans/results/model.json`
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
- NOTE: longer version of "short" for SpecAugment: 30ep->50ep

### train.pt_tc_pytorch_train_pytorch_transformer_short_bpe8000_asrtrans_mttrans (2)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train.pt_tc_pytorch_train_pytorch_transformer_short_bpe8000_asrtrans_mttrans/decode_dev5.pt_decode|**45.63**|72.3|52.0|39.3|29.9|0.995|0.995|43847|44062|
- NOTE: shorten the total number epochs when pre-training the model: 100ep->30ep

### train.pt_tc_pytorch_train_pytorch_transformer_short_bpe8000_asrtrans (3)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train.pt_tc_pytorch_train_pytorch_transformer_short_bpe8000_asrtrans/decode_dev5.pt_decode|**45.03**|71.7|51.2|38.5|29.0|1.000|1.000|44058|44062|
- NOTE: shorten the total number epochs when pre-training the model: 100ep->30ep

### train.pt_tc_pytorch_train_pytorch_transformer_ctc_asr0.3_bpe8000
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train.pt_tc_pytorch_train_pytorch_transformer_ctc_asr0.3_bpe8000/decode_dev5.pt_decode|**45.10**|71.7|51.2|38.5|29.2|1.000|1.001|44109|44062|

### train.pt_tc_pytorch_train_pytorch_transformer_ctc_asr0.2_mt0.2_bpe8000
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train.pt_tc_pytorch_train_pytorch_transformer_ctc_asr0.2_mt0.2_bpe8000/decode_dev5.pt_decode|**44.90**|71.9|51.3|38.4|29.0|0.998|0.998|43958|44062|

### train.pt_tc_pytorch_train_pytorch_transformer_bpe8000
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train.pt_tc_pytorch_train_pytorch_transformer_bpe8000/decode_dev5.pt_decode|**40.59**|68.4|46.9|34.1|25.1|0.997|0.997|43925|44062|


# RNN results
### train.pt_tc_pytorch_train_bpe1000
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train.pt_tc_pytorch_train_bpe1000/decode_dev5.pt_decode|**37.61**|65.9|44.0|31.3|22.5|0.995|0.995|43859|44059|
