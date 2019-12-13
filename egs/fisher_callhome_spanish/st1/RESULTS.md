# NOTE: apostrophe is included both in hyp and ref

# Summary (BLEU)
|model|fisher_dev|fisher_dev2|fisher_test|callhome_devtest|callhome_evltest|
|-----|----------|-----------|-----------|----------------|----------------|
|RNN (char) [[Weiss et al.]](https://arxiv.org/abs/1703.08581)|48.30|49.10|48.70|16.80|17.40|
|RNN (char, old)|40.42|41.49|41.51|14.10|14.20|
|RNN (BPE1k)|30.96|31.56|31.31|9.74|10.30|
|RNN (BPE1k) + ASR-CTC|36.54|36.99|35.57|12.19|12.66|
|Transformer (char) + ASR-CTC|45.51|46.64|45.61|17.10|16.60|
|Transformer (BPE1k) + ASR-CTC|45.85|47.73|45.77|16.99|16.78|
|Transformer (BPE1k) + ASR-CTC + MT|46.98|47.95|46.87|17.88|17.65|
|Transformer (BPE1k) + ASR-PT|46.93|47.87|47.01|17.03|17.22|
|Transformer (BPE1k) + ASR-PT + MT-PT|48.53|48.91|48.24|17.67|17.66|
|Transformer (BPE1k) + ASR-PT + MT-PT + SpecAugment|**49.64**|**50.55**|**49.22**|**18.91**|**18.87**|

### RNN (lc, character unit)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc_pytorch_train/decode_fisher_dev.en_decode|**40.42**|71.4|49.0|33.6|22.7|1.000|1.018|40695|39981|
|exp/train_sp.en_lc_pytorch_train/decode_fisher_dev2.en_decode|**41.49**|71.9|49.9|34.8|23.8|1.000|1.027|40285|39213|
|exp/train_sp.en_lc_pytorch_train/decode_fisher_test.en_decode|**41.51**|72.9|50.1|34.6|23.5|1.000|1.034|40358|39049|
|exp/train_sp.en_lc_pytorch_train/decode_callhome_devtest.en_decode|**14.10**|41.3|19.0|9.8|5.2|0.996|0.996|37268|37424|
|exp/train_sp.en_lc_pytorch_train/decode_callhome_evltest.en_decode|**14.20**|41.4|19.1|10.0|5.5|0.982|0.982|18139|18463|

### RNN (lc.rm, BPE1k)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc.rm_pytorch_train_rnn_spm_bpe1000/decode_fisher_dev.en_decode_rnn_spm|**30.96**|63.8|39.0|24.3|15.2|1.000|1.034|41550|40188|
|exp/train_sp.en_lc.rm_pytorch_train_rnn_spm_bpe1000/decode_fisher_dev2.en_decode_rnn_spm|**31.56**|64.2|39.6|25.1|15.5|1.000|1.044|41442|39711|
|exp/train_sp.en_lc.rm_pytorch_train_rnn_spm_bpe1000/decode_fisher_test.en_decode_rnn_spm|**31.31**|65.1|39.4|24.6|15.2|1.000|1.045|41381|39614|
|exp/train_sp.en_lc.rm_pytorch_train_rnn_spm_bpe1000/decode_callhome_devtest.en_decode_rnn_spm|**9.74**|35.3|13.8|6.3|3.0|1.000|1.017|38063|37416|
|exp/train_sp.en_lc.rm_pytorch_train_rnn_spm_bpe1000/decode_callhome_evltest.en_decode_rnn_spm|**10.30**|35.2|14.2|6.7|3.4|1.000|1.018|18788|18457|

### RNN (lc.rm, BPE1k) + ASR-CTC
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc.rm_pytorch_train_rnn_spm_ctc_asr0.3_bpe1000/decode_fisher_dev.en_decode_rnn_spm|**36.54**|68.5|44.9|29.7|19.5|1.000|1.032|41512|40226|
|exp/train_sp.en_lc.rm_pytorch_train_rnn_spm_ctc_asr0.3_bpe1000/decode_fisher_dev2.en_decode_rnn_spm|**36.99**|68.6|45.3|30.2|19.9|1.000|1.042|41243|39593|
|exp/train_sp.en_lc.rm_pytorch_train_rnn_spm_ctc_asr0.3_bpe1000/decode_fisher_test.en_decode_rnn_spm|**35.57**|68.8|44.1|28.6|18.4|1.000|1.050|41540|39565|
|exp/train_sp.en_lc.rm_pytorch_train_rnn_spm_ctc_asr0.3_bpe1000/decode_callhome_devtest.en_decode_rnn_spm|**12.19**|39.3|16.8|8.2|4.1|1.000|1.017|38052|37416|
|exp/train_sp.en_lc.rm_pytorch_train_rnn_spm_ctc_asr0.3_bpe1000/decode_callhome_evltest.en_decode_rnn_spm|**12.66**|39.0|17.1|8.5|4.5|1.000|1.005|18557|18457|

### Transformer (lc.rm, character unit v2) + ASR-CTC (NOTE: Transformer ST does not converge w/o ASR-CTC on this corpus)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_char_ctc_asr0.3_bpe53/decode_fisher_dev.en_decode_pytorch_transformer_char|45.51|75.8|54.3|38.6|27.1|1.000|1.016|40943|40279|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_char_ctc_asr0.3_bpe53/decode_fisher_dev2.en_decode_pytorch_transformer_char|46.64|76.6|55.3|39.6|28.2|1.000|1.018|40233|39508|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_char_ctc_asr0.3_bpe53/decode_fisher_test.en_decode_pytorch_transformer_char|45.61|77.0|54.7|38.4|26.7|1.000|1.026|40451|39441|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_char_ctc_asr0.3_bpe53/decode_callhome_devtest.en_decode_pytorch_transformer_char|17.10|45.8|22.6|12.3|6.7|1.000|1.008|37717|37416|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_char_ctc_asr0.3_bpe53/decode_callhome_evltest.en_decode_pytorch_transformer_char|16.60|45.3|22.0|11.7|6.5|1.000|1.005|18557|18457|

### Transformer (lc.rm, BPE1k) + ASR-CTC (NOTE: Transformer ST does not converge w/o ASR-CTC on this corpus)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_ctc_asr0.3_bpe1000/decode_fisher_dev.en_decode_pytorch_transformer_bpe|**45.85**|76.0|54.5|38.9|27.4|1.000|1.006|40018|39786|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_ctc_asr0.3_bpe1000/decode_fisher_dev2.en_decode_pytorch_transformer_bpe|**47.73**|77.1|56.4|41.0|29.1|1.000|1.009|39443|39089|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_ctc_asr0.3_bpe1000/decode_fisher_test.en_decode_pytorch_transformer_bpe|**45.77**|77.1|54.7|38.6|26.9|1.000|1.015|39590|38993|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_ctc_asr0.3_bpe1000/decode_callhome_devtest.en_decode_pytorch_transformer_bpe|**16.99**|45.7|22.5|12.2|6.8|0.996|0.996|37258|37416|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_ctc_asr0.3_bpe1000/decode_callhome_evltest.en_decode_pytorch_transformer_bpe|**16.78**|45.6|22.4|12.1|7.0|0.980|0.980|18086|18457|

### Transformer (lc.rm, BPE1k) + ASR-CTC + MT
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_ctc_asr0.2_mt0.2_bpe1000/decode_fisher_dev.en_decode_pytorch_transformer_bpe|**46.98**|76.7|55.8|40.0|28.4|1.000|1.006|39978|39722|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_ctc_asr0.2_mt0.2_bpe1000/decode_fisher_dev2.en_decode_pytorch_transformer_bpe|**47.95**|77.2|56.6|41.1|29.4|1.000|1.012|39640|39183|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_ctc_asr0.2_mt0.2_bpe1000/decode_fisher_test.en_decode_pytorch_transformer_bpe|**46.87**|78.0|55.9|39.8|27.8|1.000|1.015|39648|39066|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_ctc_asr0.2_mt0.2_bpe1000/decode_callhome_devtest.en_decode_pytorch_transformer_bpe|**17.88**|46.5|23.5|12.9|7.3|1.000|1.000|37414|37416|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_ctc_asr0.2_mt0.2_bpe1000/decode_callhome_evltest.en_decode_pytorch_transformer_bpe|**17.65**|46.5|23.1|12.8|7.4|0.989|0.989|18250|18457|

### Transformer (lc.rm, BPE1k) + ASR pre-training
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_bpe1000_asrtrans/decode_fisher_dev.en_decode_pytorch_transformer_bpe|**46.93**|77.0|55.7|39.9|28.3|1.000|1.001|39569|39538|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_bpe1000_asrtrans/decode_fisher_dev2.en_decode_pytorch_transformer_bpe|**47.87**|77.5|56.6|41.1|29.1|1.000|1.009|39330|38964|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_bpe1000_asrtrans/decode_fisher_test.en_decode_pytorch_transformer_bpe|**47.01**|78.1|55.9|39.8|28.1|1.000|1.014|39408|38881|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_bpe1000_asrtrans/decode_callhome_devtest.en_decode_pytorch_transformer_bpe|**17.03**|46.2|22.9|12.4|6.9|0.982|0.982|36755|37416|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_bpe1000_asrtrans/decode_callhome_evltest.en_decode_pytorch_transformer_bpe|**17.22**|46.4|23.2|12.8|7.2|0.970|0.970|17905|18457|

- Model files (archived to train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_bpe1000_asrtrans.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1b13bYQnLSkloRANflicRfsovL3-tqpX6
  - training config file: `conf/tuning/train_pytorch_transformer_bpe_short_long.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer_bpe.yaml`
  - cmvn file: `data/train_sp.en/cmvn.ark`
  - e2e file: `exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_bpe1000_asrtrans/results/model.val5.avg.best`
  - e2e JSON file: `exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_bpe1000_asrtrans/results/model.json`
  - NOTE: This model is initialized with the Transformer ASR model (BPE1k) on the encoder side.
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)

### Transformer (lc.rm, BPE1k) + ASR pre-training + MT pre-training
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_bpe1000_asrtrans_mttrans/decode_fisher_dev.en_decode_pytorch_transformer_bpe|**48.53**|77.7|57.0|41.7|30.1|1.000|1.001|39566|39509|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_bpe1000_asrtrans_mttrans/decode_fisher_dev2.en_decode_pytorch_transformer_bpe|**48.91**|78.4|57.7|42.1|30.1|1.000|1.005|39010|38828|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_bpe1000_asrtrans_mttrans/decode_fisher_test.en_decode_pytorch_transformer_bpe|**48.24**|78.7|57.0|41.1|29.4|1.000|1.013|39370|38862|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_bpe1000_asrtrans_mttrans/decode_callhome_devtest.en_decode_pytorch_transformer_bpe|**17.67**|47.1|23.7|13.1|7.4|0.973|0.973|36403|37416|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_bpe1000_asrtrans_mttrans/decode_callhome_evltest.en_decode_pytorch_transformer_bpe|**17.66**|46.7|23.6|13.3|7.6|0.966|0.966|17836|18457|

- Model files (archived to train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_bpe1000_asrtrans_mttrans.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1rQGmW0zXXzvJdyxQLKyWCG8WEkfKnPO6
  - training config file: `conf/tuning/train_pytorch_transformer_bpe_short_long.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer_bpe.yaml`
  - cmvn file: `data/train_sp.en/cmvn.ark`
  - e2e file: `exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_bpe1000_asrtrans_mttrans/results/model.val5.avg.best`
  - e2e JSON file: `exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_bpe1000_asrtrans_mttrans/results/model.json`
  - NOTE: This model is initialized with the Transformer ASR model (BPE1k) on the encoder side and Transformer MT model (BPE1k) on the decoder side.
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)

### Transformer (lc.rm, BPE1k) + ASR pre-training + MT pre-training + SpecAugment
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_long_bpe1000_specaug_asrtrans_mttrans/decode_fisher_dev.en_decode_pytorch_transformer_bpe|**49.64**|78.8|58.4|42.8|30.8|1.000|1.001|39672|39647|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_long_bpe1000_specaug_asrtrans_mttrans/decode_fisher_dev2.en_decode_pytorch_transformer_bpe|**50.55**|79.1|59.2|43.8|31.8|1.000|1.005|39216|39039|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_long_bpe1000_specaug_asrtrans_mttrans/decode_fisher_test.en_decode_pytorch_transformer_bpe|**49.22**|79.5|58.2|42.2|30.1|1.000|1.016|39494|38879|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_long_bpe1000_specaug_asrtrans_mttrans/decode_callhome_devtest.en_decode_pytorch_transformer_bpe|**18.91**|48.4|25.2|14.3|8.3|0.971|0.972|36357|37416|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_long_bpe1000_specaug_asrtrans_mttrans/decode_callhome_evltest.en_decode_pytorch_transformer_bpe|**18.87**|48.1|24.9|14.2|8.4|0.971|0.972|17938|18457|

- Model files (archived to train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_long_bpe1000_specaug_asrtrans_mttrans.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1hawp5ZLw4_SIHIT3edglxbKIIkPVe8n3
  - training config file: `conf/tuning/train_pytorch_transformer_bpe_short_long.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer_bpe.yaml`
  - preprocess config file: `conf/specaug.yaml`
  - cmvn file: `data/train_sp.en/cmvn.ark`
  - e2e file: `exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_long_bpe1000_specaug_asrtrans_mttrans/results/model.val5.avg.best`
  - e2e JSON file: `exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_short_long_bpe1000_specaug_asrtrans_mttrans/results/model.json`
  - NOTE: This model is initialized with the Transformer ASR model (BPE1k, use SpecAugment) on the encoder side and Transformer MT model (BPE1k) on the decoder side.
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
