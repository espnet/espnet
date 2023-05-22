# E-Branchformer

- ASR config: [conf/tuning/train_asr_e_branchformer_e16.yaml](conf/tuning/train_asr_e_branchformer_e16.yaml)
- Params: 43.16M
- Model link: [https://huggingface.co/pyf98/fisher_callhome_spanish_e_branchformer](https://huggingface.co/pyf98/fisher_callhome_spanish_e_branchformer)

## Environments
- date: `Tue Feb 28 21:03:54 CST 2023`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202301`
- pytorch version: `pytorch 1.13.1`
- Git hash: `568bd0808f7509f9735282537db4c68dc3bdf376`
  - Commit date: `Tue Feb 28 06:06:06 2023 -0500`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_conformer_asr_model_valid.acc.ave/callhome_devtest|3964|37989|69.4|23.2|7.5|7.2|37.8|79.0|
|decode_conformer_asr_model_valid.acc.ave/callhome_evltest|1829|19035|68.6|23.1|8.2|6.3|37.6|81.7|
|decode_conformer_asr_model_valid.acc.ave/fisher_dev|3979|40961|83.7|11.9|4.5|4.2|20.5|62.4|
|decode_conformer_asr_model_valid.acc.ave/fisher_dev2|3961|39888|84.4|11.8|3.9|4.6|20.2|62.8|
|decode_conformer_asr_model_valid.acc.ave/fisher_test|3641|40011|86.3|10.2|3.5|5.0|18.7|60.3|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_conformer_asr_model_valid.acc.ave/callhome_devtest|3964|181052|84.4|6.4|9.3|6.4|22.0|79.0|
|decode_conformer_asr_model_valid.acc.ave/callhome_evltest|1829|91266|83.7|6.4|9.8|5.7|21.9|81.7|
|decode_conformer_asr_model_valid.acc.ave/fisher_dev|3979|194297|93.2|2.6|4.2|3.9|10.7|62.4|
|decode_conformer_asr_model_valid.acc.ave/fisher_dev2|3961|189965|93.8|2.5|3.6|4.1|10.3|62.8|
|decode_conformer_asr_model_valid.acc.ave/fisher_test|3641|194507|94.8|2.1|3.1|4.6|9.8|60.3|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_conformer_asr_model_valid.acc.ave/callhome_devtest|3964|57692|66.2|18.6|15.2|5.0|38.8|79.0|
|decode_conformer_asr_model_valid.acc.ave/callhome_evltest|1829|28951|65.5|18.2|16.3|4.8|39.3|81.7|
|decode_conformer_asr_model_valid.acc.ave/fisher_dev|3979|55907|83.4|9.7|6.9|3.9|20.5|62.4|
|decode_conformer_asr_model_valid.acc.ave/fisher_dev2|3961|53966|84.5|9.6|5.8|4.2|19.6|62.8|
|decode_conformer_asr_model_valid.acc.ave/fisher_test|3641|54212|86.8|8.2|5.0|4.8|18.0|60.3|



# Conformer

- ASR config: [conf/tuning/train_asr_conformer6.yaml](conf/tuning/train_asr_conformer6.yaml)
- Params: 43.76M
- Model link: [https://huggingface.co/pyf98/fisher_callhome_spanish_conformer](https://huggingface.co/pyf98/fisher_callhome_spanish_conformer)

## Environments
- date: `Tue Feb 28 20:50:34 CST 2023`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202301`
- pytorch version: `pytorch 1.13.1`
- Git hash: `568bd0808f7509f9735282537db4c68dc3bdf376`
  - Commit date: `Tue Feb 28 06:06:06 2023 -0500`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_conformer_asr_model_valid.acc.ave/callhome_devtest|3964|37989|68.2|23.8|7.9|6.5|38.3|79.2|
|decode_conformer_asr_model_valid.acc.ave/callhome_evltest|1829|19035|67.5|24.0|8.5|6.3|38.8|82.4|
|decode_conformer_asr_model_valid.acc.ave/fisher_dev|3979|40961|83.3|12.0|4.6|4.0|20.7|63.2|
|decode_conformer_asr_model_valid.acc.ave/fisher_dev2|3961|39888|83.7|12.1|4.1|4.7|20.9|63.2|
|decode_conformer_asr_model_valid.acc.ave/fisher_test|3641|40011|85.7|10.7|3.6|5.2|19.4|61.5|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_conformer_asr_model_valid.acc.ave/callhome_devtest|3964|181052|83.6|6.7|9.7|6.0|22.4|79.2|
|decode_conformer_asr_model_valid.acc.ave/callhome_evltest|1829|91266|83.1|6.8|10.1|5.7|22.6|82.4|
|decode_conformer_asr_model_valid.acc.ave/fisher_dev|3979|194297|93.0|2.7|4.3|3.9|10.9|63.2|
|decode_conformer_asr_model_valid.acc.ave/fisher_dev2|3961|189965|93.5|2.7|3.9|4.2|10.7|63.2|
|decode_conformer_asr_model_valid.acc.ave/fisher_test|3641|194507|94.6|2.2|3.2|4.7|10.1|61.5|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_conformer_asr_model_valid.acc.ave/callhome_devtest|3964|57692|65.2|19.2|15.6|4.6|39.4|79.2|
|decode_conformer_asr_model_valid.acc.ave/callhome_evltest|1829|28951|64.3|19.0|16.7|4.9|40.5|82.4|
|decode_conformer_asr_model_valid.acc.ave/fisher_dev|3979|55907|83.1|9.8|7.1|3.8|20.7|63.2|
|decode_conformer_asr_model_valid.acc.ave/fisher_dev2|3961|53966|83.8|10.0|6.2|4.3|20.4|63.2|
|decode_conformer_asr_model_valid.acc.ave/fisher_test|3641|54212|86.4|8.6|5.0|4.9|18.5|61.5|



## Environments
- date: `Fri Feb 25 11:45:29 EST 2022`
- python version: `3.9.7 (default, Sep 16 2021, 13:09:58)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.7a1`
- pytorch version: `pytorch 1.9.0`
- Git hash: `54799d2fa7beb702ab909a7e57cc70288e3ce96c`
  - Commit date: `Tue Feb 22 10:31:31 2022 -0500`

## asr_8k_conformer (no callhome training)
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/callhome_devtest|3956|37982|64.9|24.8|10.3|6.4|41.5|79.8|
|decode_asr_asr_model_valid.acc.ave/callhome_evltest|1825|19035|63.1|25.6|11.3|6.4|43.3|82.2|
|decode_asr_asr_model_valid.acc.ave/fisher_dev|3977|40961|78.5|13.4|8.1|4.8|26.3|65.6|
|decode_asr_asr_model_valid.acc.ave/fisher_dev2|3958|39871|78.2|14.0|7.8|5.8|27.7|68.1|
|decode_asr_asr_model_valid.acc.ave/fisher_test|3641|40011|80.0|12.8|7.2|5.8|25.8|64.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/callhome_devtest|3956|180997|80.8|7.1|12.1|6.2|25.4|79.8|
|decode_asr_asr_model_valid.acc.ave/callhome_evltest|1825|91266|79.2|7.6|13.2|5.9|26.8|82.2|
|decode_asr_asr_model_valid.acc.ave/fisher_dev|3977|194297|88.6|3.4|8.0|5.3|16.7|65.6|
|decode_asr_asr_model_valid.acc.ave/fisher_dev2|3958|189893|88.4|3.8|7.7|7.0|18.6|68.1|
|decode_asr_asr_model_valid.acc.ave/fisher_test|3641|194507|89.6|3.2|7.3|5.9|16.3|64.2|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/callhome_devtest|3956|56665|64.3|20.7|15.0|5.6|41.3|79.8|
|decode_asr_asr_model_valid.acc.ave/callhome_evltest|1825|28386|62.2|21.4|16.3|6.1|43.9|82.2|
|decode_asr_asr_model_valid.acc.ave/fisher_dev|3977|55856|79.0|11.6|9.4|6.5|27.5|65.6|
|decode_asr_asr_model_valid.acc.ave/fisher_dev2|3958|53962|79.1|12.5|8.4|8.9|29.8|68.1|
|decode_asr_asr_model_valid.acc.ave/fisher_test|3641|54138|81.4|10.7|7.9|7.7|26.3|64.2|

## asr_8k_transformer (no callhome training)
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/callhome_devtest|3956|37982|53.1|33.3|13.7|6.0|52.9|85.1|
|decode_asr_asr_model_valid.acc.ave/callhome_evltest|1825|19035|52.3|34.0|13.7|6.0|53.7|86.7|
|decode_asr_asr_model_valid.acc.ave/fisher_dev|3977|40961|76.8|16.5|6.7|5.1|28.3|70.0|
|decode_asr_asr_model_valid.acc.ave/fisher_dev2|3958|39871|77.8|16.3|5.9|6.0|28.2|70.8|
|decode_asr_asr_model_valid.acc.ave/fisher_test|3641|40011|79.9|14.5|5.5|5.8|25.9|69.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/callhome_devtest|3956|180997|74.3|10.0|15.7|6.2|31.9|85.1|
|decode_asr_asr_model_valid.acc.ave/callhome_evltest|1825|91266|73.3|10.2|16.5|6.1|32.8|86.7|
|decode_asr_asr_model_valid.acc.ave/fisher_dev|3977|194297|89.7|4.0|6.3|5.6|15.9|70.0|
|decode_asr_asr_model_valid.acc.ave/fisher_dev2|3958|189893|90.4|4.1|5.5|7.0|16.6|70.8|
|decode_asr_asr_model_valid.acc.ave/fisher_test|3641|194507|91.6|3.4|5.0|5.9|14.3|69.2|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/callhome_devtest|3956|56665|52.2|28.0|19.8|5.0|52.8|85.1|
|decode_asr_asr_model_valid.acc.ave/callhome_evltest|1825|28386|50.8|28.9|20.2|5.2|54.3|86.7|
|decode_asr_asr_model_valid.acc.ave/fisher_dev|3977|55856|76.2|14.3|9.5|5.8|29.6|70.0|
|decode_asr_asr_model_valid.acc.ave/fisher_dev2|3958|53962|77.5|14.3|8.2|7.9|30.4|70.8|
|decode_asr_asr_model_valid.acc.ave/fisher_test|3641|54138|80.1|12.3|7.5|6.6|26.5|69.2|


## asr_train_asr_raw_bpe1000_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev_all|12986|81587|77.8|16.1|6.1|6.0|28.2|62.4|
|decode_asr_asr_model_valid.acc.ave/test|6283|40307|80.5|14.6|4.9|5.9|25.4|61.4|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev_all|12986|392279|89.7|3.9|6.4|5.7|16.0|62.4|
|decode_asr_asr_model_valid.acc.ave/test|6283|195370|91.8|3.3|4.9|5.6|13.9|61.4|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev_all|12986|115994|76.9|13.3|9.7|5.4|28.5|62.4|
|decode_asr_asr_model_valid.acc.ave/test|6283|55738|80.2|12.0|7.9|5.8|25.6|61.4|



## asr_train_asr_conformer6_raw_bpe1000_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev_all|12986|81587|82.4|12.4|5.2|5.4|23.0|57.5|
|decode_asr_asr_model_valid.acc.ave/test|6283|40307|85.0|11.0|4.1|5.4|20.5|55.5|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev_all|12986|392279|91.6|2.9|5.4|5.3|13.7|57.5|
|decode_asr_asr_model_valid.acc.ave/test|6283|195370|93.6|2.4|4.0|5.4|11.7|55.5|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev_all|12986|115994|81.6|10.1|8.3|5.3|23.7|57.5|
|decode_asr_asr_model_valid.acc.ave/test|6283|55738|84.9|8.6|6.5|5.7|20.7|55.5|
