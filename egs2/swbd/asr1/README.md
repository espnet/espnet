# Conformer
## Environments
- date: `Tue Dec 27 03:17:13 CST 2022`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202211`
- pytorch version: `pytorch 1.12.1`
- Git hash: `ef3ce328551c12c03284defc757f42df47c46170`
  - Commit date: `Mon Dec 26 20:34:28 2022 -0500`

## asr_train_asr_conformer_e12_size256_linear2048_raw_en_bpe2000_sp
- ASR config: [conf/tuning/train_asr_conformer_e12_size256_linear2048.yaml](conf/tuning/train_asr_conformer_e12_size256_linear2048.yaml)
- Params: 44.53M
- Model link: [https://huggingface.co/pyf98/swbd_conformer](https://huggingface.co/pyf98/swbd_conformer)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval2000/hyp.callhm.ctm.filt.sys|2628|21594|88.3|8.5|3.2|2.1|13.8|46.8|
|decode_asr_asr_model_valid.acc.ave/eval2000/hyp.ctm.filt.sys|4459|42989|90.8|6.2|3.0|1.5|10.7|42.4|
|decode_asr_asr_model_valid.acc.ave/eval2000/hyp.swbd.ctm.filt.sys|1831|21395|93.4|3.9|2.7|0.9|7.5|36.1|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval2000|4322|181952|92.8|3.4|3.8|10.8|18.0|68.1|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval2000|4322|47227|82.6|12.8|4.6|15.3|32.7|68.1|



# Conformer: initial
## Environments
- date: `Fri May 14 07:43:22 UTC 2021`
- model link: https://zenodo.org/record/4978923/files/asr_train_asr_cformer5_raw_bpe2000_sp_valid.acc.ave.zip?download=1
- python version: `3.8.8 (default, Apr 13 2021, 19:58:26)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.9`
- pytorch version: `pytorch 1.8.0+cu111`
- Git hash: `64f026d35013e9f0058bcdeab86eb28fed48ed4b`
  - Commit date: `Fri May 7 09:31:16 2021 +0000`

## asr_train_asr_cformer5_raw_bpe2000_sp
### WER

```
exp_sp/train_nodup_sp_pytorch_train_pytorch_conformer_lr5_specaug_resume/decode_eval2000_model.last10.avg.best_decode_train_transformer_lm_pytorch_swbd+fisher_bpe2000/scoring/hyp.callhm.ctm.filt.sys
|       SPKR              |        # Snt              # Wrd        |        Corr                 Sub                  Del                 Ins                  Err               S.Err        |
|       Sum/Avg           |        2628               21594        |        84.4                 9.6                  3.8                 2.2                 15.6                49.4        |
exp_sp/train_nodup_sp_pytorch_train_pytorch_conformer_lr5_specaug_resume/decode_eval2000_model.last10.avg.best_decode_train_transformer_lm_pytorch_swbd+fisher_bpe2000/scoring/hyp.ctm.filt.sys
|       SPKR              |       # Snt             # Wrd        |       Corr                 Sub                Del                 Ins                 Err              S.Err        |
|       Sum/Avg           |       4459              42989        |       89.6                7.0                3.4                 1.6                12.0               44.9        |
exp_sp/train_nodup_sp_pytorch_train_pytorch_conformer_lr5_specaug_resume/decode_eval2000_model.last10.avg.best_decode_train_transformer_lm_pytorch_swbd+fisher_bpe2000/scoring/hyp.swbd.ctm.filt.sys
|       SPKR             |        # Snt              # Wrd        |       Corr                  Sub                 Del                 Ins                  Err               S.Err        |
|       Sum/Avg          |        1831               21395        |       92.5                  4.4                 3.1                 0.9                  8.4                38.3        |
```

# Fbank-pitch Model with BPE2k, Specaug- with and without 12L, BPE2k SWBD+Fisher Transformer LM

## Environments
- date: `Sun Jun 20 20:06:18 EDT 2021`
- python version: `3.8.8 (default, Feb 24 2021, 21:46:12)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.9`
- pytorch version: `pytorch 1.8.0`
- Git hash: `72a229498b4f8f8aff5251753862668a0a955ebe`
- Commit date: `Sat Jun 19 19:42:50 2021 -0400`
- model link: https://zenodo.org/record/5006864/files/asr_train_conformer_fbank_pitch_bpe2000_valid.acc.ave_10best.zip?download=1

Without LM Rescoring
```
exp/asr_train_asr_conformer_fbank_pitch_bpe2000/decode_asr_asr_model_valid.acc.ave_10best/eval2000/score_wer/scoring/hyp.callhm.ctm.filt.sys
|       SPKR              |        # Snt              # Wrd        |        Corr                 Sub                  Del                 Ins                  Err               S.Err        |
|	Sum/Avg	|	2628	|	21594	|	87.4	|	9.6	|	3.0	|	2.0	|	14.6	|	49.7	|
exp/asr_train_asr_conformer_fbank_pitch_bpe2000/decode_asr_asr_model_valid.acc.ave_10best/eval2000/score_wer/scoring/hyp.ctm.filt.sys	
|       SPKR              |        # Snt              # Wrd        |        Corr                 Sub                  Del                 Ins                  Err               S.Err        |
|	Sum/Avg	|	4459	|	42989	|	90.5	|	7.0	|	2.5	|	1.5	|	10.9	|	44.7	|
exp/asr_train_asr_conformer_fbank_pitch_bpe2000/decode_asr_asr_model_valid.acc.ave_10best/eval2000/score_wer/scoring/hyp.swbd.ctm.filt.sys
|       SPKR              |        # Snt              # Wrd        |        Corr                 Sub                  Del                 Ins                  Err               S.Err        |
|	Sum/Avg	|	1831	|	21395	|	93.7	|	4.3	|	2.0	|	0.9	|	7.2	|	37.7	|
```
 With Transformer LM
```
exp/asr_train_asr_conformer_fbank_pitch_bpe2000/decode_lm_transformer_valid.loss.ave_asr_model_valid.acc.ave_10best/eval2000/score_wer/scoring/hyp.callhm.ctm.filt.sys
|       SPKR              |        # Snt              # Wrd        |        Corr                 Sub                  Del                 Ins                  Err               S.Err        |
|	Sum/Avg	|	2628	|	21594	|	88.0	|	8.9	|	3.1	|	2.0	|	14.0	|	48.0	|
exp/asr_train_asr_conformer_fbank_pitch_bpe2000/decode_lm_transformer_valid.loss.ave_asr_model_valid.acc.ave_10best/eval2000/score_wer/scoring/hyp.ctm.filt.sys
|       SPKR              |        # Snt              # Wrd        |        Corr                 Sub                  Del                 Ins                  Err               S.Err        |
|	Sum/Avg	|	4459	|	42989	|	91.0	|	6.5	|	2.5	|	1.4	|	10.4	|	43.0	|
exp/asr_train_asr_conformer_fbank_pitch_bpe2000/decode_lm_transformer_valid.loss.ave_asr_model_valid.acc.ave_10best/eval2000/score_wer/scoring/hyp.swbd.ctm.filt.sys
|       SPKR              |        # Snt              # Wrd        |        Corr                 Sub                  Del                 Ins                  Err               S.Err        |
|	Sum/Avg	|	1831	|	21395	|	94.0	|	4.0	|	2.0	|	0.9	|	6.8	|	35.9	|
```
