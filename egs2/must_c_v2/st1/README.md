# RESULTS

# Offline ST results (BLEU)

## Attentional Enc-Dec (st_train_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp)

- ST config: [train_st_conformer_asrinit_v2.yaml](./conf/tuning/train_st_conformer_asrinit_v2.yaml)
- Inference config: [decode_st_conformer.yaml](./conf/tuning/decode_st_conformer.yaml)
- Download model and run inference:

    `./run.sh --skip_data_prep false --skip_train true --download_model espnet/brianyan918_mustc-v2_en-de_st_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp --inference_config conf/tuning/decode_st_conformer.yaml`

|dataset|score|verbose_score|
|---|---|---|
|decode_st_conformer_st_model_valid.acc.ave_10best/tst-COMMON.en-de|25.7|62.3/34.6/21.8/14.3 (BP = 0.897 ratio = 0.902 hyp_len = 46612 ref_len = 51699)|

## Multi-Decoder Attn Enc-Dec (st_train_st_md_conformer_asrinit_v3-2_raw_en_de_bpe_tc4000_sp)

- ST config: [train_st_md_conformer_asrinit_v3-2.yaml](./conf/tuning/train_st_md_conformer_asrinit_v3-2.yaml)
- Inference config: [decode_st_md.yaml](./conf/tuning/decode_st_md.yaml)
- Download model and run inference:

    `./run.sh --skip_data_prep false --skip_train true --download_model espnet/brianyan918_mustc-v2_en-de_st_md_conformer_asrinit_v3-2_raw_en_de_bpe_tc4000_sp --inference_config conf/tuning/decode_st_md.yaml`

|dataset|score|verbose_score|
|---|---|---|
|decode_st_md_st_model_valid.acc.ave_10best/tst-COMMON.en-de|27.6|61.6/34.6/21.9/14.4 (BP = 0.964 ratio = 0.965 hyp_len = 49877 ref_len = 51699)|

## CTC/Attention (st_train_st_ctc_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp)

- ST config: [train_st_ctc_conformer_asrinit_v2.yaml](./conf/tuning/train_st_ctc_conformer_asrinit_v2.yaml)
- Inference config: [decode_st_conformer_ctc0.3.yaml](./conf/tuning/decode_st_conformer_ctc0.3.yaml)
- Download model and run inference:

    `./run.sh --skip_data_prep false --skip_train true --download_model espnet/brianyan918_mustc-v2_en-de_st_ctc_conformer_asrinit_v2_raw_en_de_bpe_tc4000_sp --inference_config conf/tuning/decode_st_conformer_ctc0.3.yaml`

|dataset|score|verbose_score|
|---|---|---|
|decode_st_conformer_ctc0.3_st_model_valid.acc.ave_10best/tst-COMMON.en-de|28.6|61.8/35.1/22.2/14.5 (BP = 0.988 ratio = 0.988 hyp_len = 51068 ref_len = 51699)|

## Multi-Decoder CTC/Attention (st_train_st_ctc_md_conformer_asrinit_v3_noamp_batch50m_ctcsamp0.1_raw_en_de_bpe_tc4000_sp)

- ST config: [train_st_ctc_md_conformer_asrinit_v3_noamp_batch50m_ctcsamp0.1.yaml](./conf/tuning/train_st_ctc_md_conformer_asrinit_v3_noamp_batch50m_ctcsamp0.1.yaml)
- Inference config: [decode_st_md_ctc0.3.yaml](./conf/tuning/decode_st_md_ctc0.3.yaml)
- Download model and run inference:

    `./run.sh --skip_data_prep false --skip_train true --download_model espnet/brianyan918_mustc-v2_en-de_st_ctc_md_conformer_asrinit_v3_raw_en_de_bpe_tc4000_sp --inference_config conf/tuning/decode_st_md_ctc0.3.yaml`

|dataset|score|verbose_score|
|---|---|---|
|decode_st_md_ctc0.3_st_model_valid.acc.ave_10best/tst-COMMON.en-de|28.8|61.5/35.0/22.2/14.7 (BP = 0.994 ratio = 0.994 hyp_len = 51386 ref_len = 51699)|

## Transducer (st_train_st_ctc_rnnt_asrinit_raw_en_de_bpe_tc4000_sp)

- ST config: [train_st_ctc_rnnt_asrinit.yaml](./conf/tuning/train_st_ctc_rnnt_asrinit.yaml)
- Inference config: [decode_rnnt_tsd_mse4_scorenormduring_beam10.yaml](./conf/tuning/decode_rnnt_tsd_mse4_scorenormduring_beam10.yaml)
- Download model and run inference:

    `./run.sh --skip_data_prep false --skip_train true --download_model espnet/brianyan918_mustc-v2_en-de_st_ctc_rnnt_asrinit_raw_en_de_bpe_tc4000_sp --inference_config conf/tuning/decode_rnnt_tsd_mse4_scorenormduring_beam10.yaml`

|dataset|score|verbose_score|
|---|---|---|
|decode_rnnt_tsd_mse4_scorenormduring_beam10_st_model_valid.loss.ave_10best/tst-COMMON.en-de|27.6|60.2/33.6/21.0/13.7 (BP = 0.998 ratio = 0.998 hyp_len = 51602 ref_len = 51699)|
