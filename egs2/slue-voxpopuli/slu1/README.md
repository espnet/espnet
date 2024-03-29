<!-- Generated by ./scripts/utils/show_asr_result.sh -->
# RESULTS

## Environments
- date: `Mon Feb 28 12:28:28 EST 2021`
- python version: `3.9.5 (default, Jun  4 2021, 12:28:51) [GCC 7.5.0]`
- espnet version: `espnet 0.10.3a2`
- pytorch version: `pytorch 1.8.1+cu102`
- Git hash: `6bf3c2a4f138d35331634d2e879bbc5c32a5266e`
  - Commit date: `Tue Feb 22 15:41:32 EST 2021`


## Using Conformer based encoder and Transformer based decoder with spectral augmentation and predicting transcript along with intent
- SLU config: [conf/tuning/train_asr_no_pretrain.yaml](conf/tuning/train_asr_no_pretrain.yaml)
- token_type: bpe

|dataset|Snt|Micro F1 (%)|Micro Label F1 (%)|
|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/devel|1742|54.7|67.6|

## With TERA SSL Pretrain
- SLU config: [conf/tuning/train_asr_tera.yaml](conf/tuning/train_asr_tera.yaml)
- token_type: bpe

|dataset|Snt|Micro F1 (%)|Micro Label F1 (%)|
|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/devel|1742|57.1|70.9|

## With VQ-APC SSL Pretrain
- SLU config: [conf/tuning/train_asr_vq_apc.yaml](conf/tuning/train_asr_vq_apc.yaml)
- token_type: bpe

|dataset|Snt|Micro F1 (%)|Micro Label F1 (%)|
|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/devel|1742|63.6|76.6|

## With Wav2Vec2 SSL Pretrain
- SLU config: [conf/tuning/train_asr_tera.yaml](conf/tuning/train_asr_tera.yaml)
- token_type: bpe

|dataset|Snt|Micro F1 (%)|Micro Label F1 (%)|
|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/devel|1742|69.5|83.3|

## With HuBERT SSL Pretrain
- SLU config: [conf/tuning/train_asr_hubert.yaml](conf/tuning/train_asr_hubert.yaml)
- token_type: bpe

|dataset|Snt|Micro F1 (%)|Micro Label F1 (%)|
|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/devel|1742|69.7|84.8|

## With WavLM SSL Pretrain
- SLU config: [conf/tuning/train_asr_wavlm.yaml](conf/tuning/train_asr_wavlm.yaml)
- token_type: bpe

|dataset|Snt|Micro F1 (%)|Micro Label F1 (%)|
|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/devel|1742|74.5|88.0|

## With Gigaspeech ASR Pretrain
- SLU config: [conf/tuning/train_asr_gigaspeech.yaml](conf/tuning/train_asr_gigaspeech.yaml)
- token_type: bpe

|dataset|Snt|Micro F1 (%)|Micro Label F1 (%)|
|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/devel|1742|73.9|86.0|

## With SPGISpeech ASR Pretrain
- SLU config: [conf/tuning/train_asr_spgispeech.yaml](conf/tuning/train_asr_spgispeech.yaml)
- token_type: bpe

|dataset|Snt|Micro F1 (%)|Micro Label F1 (%)|
|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/devel|1742|71.9|84.1|

## With SLURP SLU Pretrain
- SLU config: [conf/tuning/train_asr_slurp.yaml](conf/tuning/train_asr_slurp.yaml)
- token_type: bpe

|dataset|Snt|Micro F1 (%)|Micro Label F1 (%)|
|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/devel|1742|59.7|71.5|

## With WavLM and SLURP SLU Pretrain
- SLU config: [conf/tuning/train_asr_slurp_wavlm.yaml](conf/tuning/train_asr_slurp_wavlm.yaml)
- token_type: bpe

|dataset|Snt|Micro F1 (%)|Micro Label F1 (%)|
|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/devel|1742|75.7|87.5|

## With BERT LM Pretrain
- SLU config: [conf/tuning/train_asr_bert.yaml](conf/tuning/train_asr_bert.yaml)
- token_type: bpe
- use_transcript: true
- pretrained_model: exp/slu_train_asr_no_pretrain_raw_en_bpe1000_sp/valid.acc.ave_10best.pth:encoder:encoder
- local_data_opts: "--use_transcript true --transcript_folder exp/slu_train_asr_no_pretrain_raw_en_bpe1000_sp/decode_asr_asr_model_valid.acc.ave"

|dataset|Snt|Micro F1 (%)|Micro Label F1 (%)|
|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/devel|1742|54.5|69.2|

## With DeBERTA LM Pretrain
- SLU config: [conf/tuning/train_asr_deberta.yaml](conf/tuning/train_asr_deberta.yaml)
- token_type: bpe
- use_transcript: true
- pretrained_model: exp/slu_train_asr_no_pretrain_raw_en_bpe1000_sp/valid.acc.ave_10best.pth:encoder:encoder
- local_data_opts: "--use_transcript true --transcript_folder exp/slu_train_asr_no_pretrain_raw_en_bpe1000_sp/decode_asr_asr_model_valid.acc.ave"

|dataset|Snt|Micro F1 (%)|Micro Label F1 (%)|
|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/devel|1742|55.5|69.4|

## With WavLM and BERT LM Pretrain
- SLU config: [conf/tuning/train_asr_bert_wavlm.yaml](conf/tuning/train_asr_bert_wavlm.yaml)
- token_type: bpe
- use_transcript: true
- pretrained_model: exp/slu_train_asr_wavlm_raw_en_bpe1000_sp/valid.acc.ave_10best.pth:encoder:encoder
- local_data_opts: "--use_transcript true --transcript_folder exp/slu_train_asr_wavlm_raw_en_bpe1000_sp/decode_asr_asr_model_valid.acc.ave"

|dataset|Snt|Micro F1 (%)|Micro Label F1 (%)|
|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/devel|1742|73.5|87.3|

## With WavLM and DeBERTA LM Pretrain
- SLU config: [conf/tuning/train_asr_deberta_wavlm.yaml](conf/tuning/train_asr_deberta_wavlm.yaml)
- token_type: bpe
- use_transcript: true
- pretrained_model: exp/slu_train_asr_wavlm_raw_en_bpe1000_sp/valid.acc.ave_10best.pth:encoder:encoder
- local_data_opts: "--use_transcript true --transcript_folder exp/slu_train_asr_wavlm_raw_en_bpe1000_sp/decode_asr_asr_model_valid.acc.ave"

|dataset|Snt|Micro F1 (%)|Micro Label F1 (%)|
|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/devel|1742|74.0|87.7|
