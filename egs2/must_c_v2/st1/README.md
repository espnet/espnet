# RESULTS

# Offline ST results (BLEU)

## CTC/Attention (st_train_st_ctc_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp)

- Model link: [huggingface]()
- ST config: [./conf/tuning/train_ctc_conformer_asrinit_v2.yaml](./conf/tuning/train_ctc_conformer_asrinit_v2.yaml)
- Inference config: [./conf/decode_st_conformer_ctc0.3.yaml](./conf/tuning/decode_st_conformer_ctc0.3.yaml)

|dataset|score|verbose_score|
|---|---|---|
|decode_st_conformer_ctc0.3_st_model_valid.acc.ave_10best/tst-COMMON.en-de|28.6|61.8/35.1/22.2/14.5 (BP = 0.988 ratio = 0.988 hyp_len = 51068 ref_len = 51699)|

## Transducer (st_train_st_ctc_rnnt_asrinit_raw_en_de_bpe_tc4000_sp)

- Model link: [huggingface]()
- ST config: [./conf/tuning/train_st_ctc_rnnt_asrinit.yaml](./conf/tuning/train_st_ctc_rnnt_asrinit.yaml)
- Inference config: [./conf/decode_rnnt_tsd_mse4_scorenormduring_beam10.yaml](./conf/tuning/decode_rnnt_tsd_mse4_scorenormduring_beam10.yaml)

|dataset|score|verbose_score|
|---|---|---|
|decode_rnnt_tsd_mse4_scorenormduring_beam10_st_model_valid.loss.ave_10best/tst-COMMON.en-de|27.6|60.2/33.6/21.0/13.7 (BP = 0.998 ratio = 0.998 hyp_len = 51602 ref_len = 51699)|