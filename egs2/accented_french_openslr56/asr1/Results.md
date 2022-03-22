# RESULTS
## Environments
- date: `Mon Mar 21 16:22:32 EDT 2022`
- python version: `3.9.7 (default, Sep 16 2021, 13:09:58)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.6a1`
- pytorch version: `pytorch 1.10.2+cu102`
- Git hash: `c2bec501b9754ef0e27c63266f9ca2d928b6de4f`
  - Commit date: `Mon Mar 21 16:05:52 2022 -0400`

## asr_baseline
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/dev|986|7455|44.2|37.5|18.2|4.7|60.5|83.8|
|decode_asr_asr_model_valid.acc.ave_10best/test|515|2941|86.3|12.6|1.1|9.1|22.8|54.4|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/dev|986|40113|62.3|13.9|23.8|5.9|43.6|83.8|
|decode_asr_asr_model_valid.acc.ave_10best/test|515|16233|96.4|2.0|1.6|2.0|5.6|54.4|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/dev|986|19017|46.5|27.3|26.2|4.0|57.5|83.8|
|decode_asr_asr_model_valid.acc.ave_10best/test|515|7998|89.6|6.6|3.8|1.2|11.7|54.4|


## asr_hubert_frontend
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/dev|986|7455|66.9|24.2|8.9|5.8|38.9|72.8|
|decode_asr_asr_model_valid.acc.ave_10best/test|515|2941|81.5|15.4|3.2|8.7|27.3|60.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/dev|986|40113|82.6|6.3|11.1|6.4|23.8|72.8|
|decode_asr_asr_model_valid.acc.ave_10best/test|515|16233|92.6|3.2|4.2|2.2|9.6|60.2|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/dev|986|19017|72.5|13.7|13.8|4.3|31.8|72.8|
|decode_asr_asr_model_valid.acc.ave_10best/test|515|7998|85.7|8.6|5.7|1.4|15.7|60.2|



## asr_frontend_linear_fusion
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/dev|986|7455|63.3|26.1|10.5|5.5|42.2|74.8|
|decode_asr_asr_model_valid.acc.ave_10best/test|515|2941|81.6|14.8|3.6|8.2|26.7|60.4|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/dev|986|40113|79.0|7.6|13.3|6.5|27.4|74.8|
|decode_asr_asr_model_valid.acc.ave_10best/test|515|16233|92.4|2.9|4.6|2.0|9.5|60.4|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/dev|986|19017|68.4|15.3|16.3|4.5|36.1|74.8|
|decode_asr_asr_model_valid.acc.ave_10best/test|515|7998|85.4|8.2|6.4|1.1|15.7|60.4|

