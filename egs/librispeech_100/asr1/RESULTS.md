# Conformer-CTC
  - Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: (<https://drive.google.com/file/d/1w-GzALrVIbCNiMpGh3UajhvXpOGPMil_>)
    - training config file: `conf/tuning/train_conformer_ctc.yaml`
    - decoding config file: `conf/tuning/decode_ctc.yaml`
    - cmvn file: `data/train_clean_100_sp/cmvn.ark`
    - e2e file: `exp/train_clean_100_sp_pytorch_train_conformer_ctc_nbpe300_specaug/results/model.cer5.avg.best`
    - e2e JSON file: `exp/train_clean_100_sp_pytorch_train_conformer_ctc_nbpe300_specaug/results/model.json`
    - dict file: `data/lang_char`

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_clean_model.cer5.avg.best_decode_ctc_nolm|2703|54402|93.3|6.1|0.6|0.7|7.4|62.6|
|decode_dev_other_model.cer5.avg.best_decode_ctc_nolm|2864|50948|81.9|16.3|1.8|2.1|20.2|84.4|
|decode_test_clean_model.cer5.avg.best_decode_ctc_nolm|2620|52576|93.1|6.3|0.6|0.8|7.7|63.5|
|decode_test_other_model.cer5.avg.best_decode_ctc_nolm|2939|52343|81.3|16.5|2.1|2.0|20.6|85.2|

# Conformer-CTC/Attention
  - Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: (<https://drive.google.com/file/d/1QTVqk4sPSdECjuqEjr7vP0ZG7PoLgIMc>)
    - training config file: `conf/tuning/train_conformer_ctcatt.yaml`
    - decoding config file: `conf/tuning/decode_ctcatt.yaml`
    - cmvn file: `data/train_clean_100_sp/cmvn.ark`
    - e2e file: `exp/train_clean_100_sp_pytorch_train_conformer_ctcatt_nbpe300_specaug/results/model.val5.avg.best`
    - e2e JSON file: `exp/train_clean_100_sp_pytorch_train_conformer_ctcatt_nbpe300_specaug/results/model.json`
    - dict file: `data/lang_char`

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_clean_model.val5.avg.best_decode_ctcatt_cw0.3_nolm|2703|54402|93.7|5.3|1.0|0.9|7.2|56.9|
|decode_dev_other_model.val5.avg.best_decode_ctcatt_cw0.3_nolm|2864|50948|83.5|14.6|1.9|2.0|18.5|81.5|
|decode_test_clean_model.val5.avg.best_decode_ctcatt_cw0.3_nolm|2620|52576|93.4|5.5|1.1|0.7|7.3|57.7|
|decode_test_other_model.val5.avg.best_decode_ctcatt_cw0.3_nolm|2939|52343|82.7|15.0|2.2|2.1|19.3|81.8|

# Conformer-Transducer
  - Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: (<https://drive.google.com/file/d/1seIYIpMe2gVYM-bWbRbrrqjFuptr1jYb>)
    - training config file: `conf/tuning/train_conformer_transducer.yaml`
    - decoding config file: `conf/tuning/decode_transducer.yaml`
    - cmvn file: `data/train_clean_100_sp/cmvn.ark`
tar: Removing leading `/' from member names
    - e2e file: `exp/train_clean_100_sp_pytorch_train_conformer_transducer_nbpe300_specaug/results/model.last10.avg.best`
    - e2e JSON file: `exp/train_clean_100_sp_pytorch_train_conformer_transducer_nbpe300_specaug/results/model.json`
tar: Removing leading `/' from member names
    - dict file: `data/lang_char`

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_clean_model.last10.avg.best_decode_transducer_nolm|2703|54402|93.4|5.9|0.7|0.7|7.3|61.6|
|decode_dev_other_model.last10.avg.best_decode_transducer_nolm|2864|50948|82.1|15.6|2.3|1.9|19.9|84.0|
|decode_test_clean_model.last10.avg.best_decode_transducer_nolm|2620|52576|93.0|6.1|0.9|0.8|7.8|63.6|
|decode_test_other_model.last10.avg.best_decode_transducer_nolm|2939|52343|82.1|15.4|2.5|1.8|19.8|84.8|
