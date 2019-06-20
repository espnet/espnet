# RESULTS
## Environments
- date: `Tue Jun 18 15:56:02 JST 2019`
- system information: `Linux SMP Thu Nov 29 14:49:43 UTC 2018 x86_64 x86_64 x86_64 GNU/Linux`
- python version: `3.6.8 |Anaconda, Inc.| (default, Dec 30 2018, 01:22:34)  [GCC 7.3.0]`
- espnet version: `espnet 0.3.1`
- chainer version: `chainer 5.0.0`
- pytorch version: `pytorch 1.1.0`
- Git hash: `88c81722113bb83a20128c38eceeb951c2d7964e`
  - Commit date: `Sat May 25 06:55:17 2019 -0400`

## Baseline(egs/asr1) 2mic
### CER

|dataset|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|
|dt05_real | 1640|160390 | 94.8| 2.7| 2.5| 1.4|6.6|67.9  |
|dt05_simu | 1640|160400 | 93.9| 3.3| 2.9| 1.5|7.6|68.6  |
|et05_real | 1320|126796 | 89.1| 6.0| 4.9| 2.8|13.7 | 79.2  |
|et05_simu | 1320|126812 | 88.9| 6.0| 5.2| 2.8|14.0 | 80.2  |
### WER

|dataset|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|
|dt05_real | 1640|27119  | 89.0| 8.9| 2.1| 1.7|12.6 | 67.9  |
|dt05_simu | 1640|27120  | 87.7| 10.1|2.2| 1.8|14.1 | 68.6  |
|et05_real | 1320|21409  | 78.8| 17.7|3.5| 3.2|24.4 | 79.2  |
|et05_simu | 1320|21416  | 78.8| 17.6|3.6| 3.3|24.5 | 80.2  |

## Baseline(egs/asr1) 5mic
### CER

|dataset|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|
|dt05_real | 1640|160390 | 96.0| 2.0| 2.0| 1.1|5.1|62.1  |
|dt05_simu | 1640|160400 | 95.3| 2.3| 2.3| 1.1|5.8|64.3  |
|et05_real | 1320|126796 | 91.4| 4.6| 3.9| 2.2|10.7|75.7  |
|et05_simu | 1320|126812 | 90.9| 4.8| 4.3| 2.3|11.4|78.3  |

### WER
|dataset|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|
|dt05_real | 1640|27119  | 91.2| 7.2| 1.7| 1.2|10.1|62.1  |
|dt05_simu | 1640|27120  | 90.2| 8.0| 1.8| 1.3|11.1|64.3  |
|et05_real | 1320|21409  | 82.8| 14.3|2.9| 2.5|19.8|75.7  |
|et05_simu | 1320|21416  | 82.2| 14.6|3.2| 2.9|20.6|78.3  |


## DNN MVDR 5mic
### CER

|dataset|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|
|decode_dt05_real_isolated_6ch_track_multich_decode_lm_word65000|1640|160390|96.5|1.7|1.8|0.8|4.4|55.6|
|decode_dt05_simu_isolated_6ch_track_multich_decode_lm_word65000|1640|160400|96.6|1.7|1.7|0.7|4.1|53.7|
|decode_et05_real_isolated_6ch_track_multich_decode_lm_word65000|1320|126796|92.8|3.5|3.6|1.7|8.9|68.9|
|decode_et05_simu_isolated_6ch_track_multich_decode_lm_word65000|1320|126812|94.2|2.8|3.0|1.2|7.1|64.5|

### WER

|dataset|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|
|decode_dt05_real_isolated_6ch_track_multich_decode_lm_word65000|1640|27119|92.2|6.4|1.5|0.9|8.8|55.6|
|decode_dt05_simu_isolated_6ch_track_multich_decode_lm_word65000|1640|27120|92.6|6.2|1.3|0.8|8.3|53.7|
|decode_et05_real_isolated_6ch_track_multich_decode_lm_word65000|1320|21409|85.6|11.6|2.8|1.9|16.3|68.9|
|decode_et05_simu_isolated_6ch_track_multich_decode_lm_word65000|1320|21416|87.9|9.9|2.2|1.4|13.4|64.5|


# Evaluation for ehnaced speech
## Noisy

### SDR

|dataset|PED|CAF|STR|BUS|MEAN|
|---|---|---|---|---|---|
|dt05_simu_isolated_6ch_track|8.468|6.848|4.449|2.961|5.68157|
|et05_simu_isolated_6ch_track|8.326|7.745|6.56|6.754|7.34641|
### STOI

|dataset|PED|CAF|STR|BUS|MEAN|
|---|---|---|---|---|---|
|dt05_simu_isolated_6ch_track|0.6629|0.6272|0.6375|0.6465|0.643516|
|et05_simu_isolated_6ch_track|0.6143|0.6016|0.6056|0.6237|0.61128|
### ESTOI

|dataset|PED|CAF|STR|BUS|MEAN|
|---|---|---|---|---|---|
|dt05_simu_isolated_6ch_track|0.5321|0.4726|0.4828|0.5009|0.497096|
|et05_simu_isolated_6ch_track|0.4765|0.4567|0.4614|0.4926|0.471788|
### Wideband PESQ

|dataset|PED|CAF|STR|BUS|MEAN|
|---|---|---|---|---|---|
|dt05_simu_isolated_6ch_track|1.368|1.217|1.215|1.278|1.26935|
|et05_simu_isolated_6ch_track|1.26|1.242|1.276|1.315|1.27316|

## DNN MVDR 5mic
### SDR

|dataset|PED|CAF|STR|BUS|MEAN|
|---|---|---|---|---|---|
|dt05_simu_isolated_6ch_track_multich|15.54|13.91|13.49|14.38|14.3307|
|et05_simu_isolated_6ch_track_multich|16.05|15.64|14.96|16.85|15.8768|
### STOI

|dataset|PED|CAF|STR|BUS|MEAN|
|---|---|---|---|---|---|
|dt05_simu_isolated_6ch_track_multich|0.7021|0.6835|0.6965|0.7095|0.697908|
|et05_simu_isolated_6ch_track_multich|0.6589|0.6568|0.6568|0.6695|0.660504|
### ESTOI

|dataset|PED|CAF|STR|BUS|MEAN|
|---|---|---|---|---|---|
|dt05_simu_isolated_6ch_track_multich|0.6096|0.5699|0.5962|0.6268|0.60064|
|et05_simu_isolated_6ch_track_multich|0.5636|0.5589|0.5606|0.5858|0.567224|
### Wideband PESQ

|dataset|PED|CAF|STR|BUS|MEAN|
|---|---|---|---|---|---|
|dt05_simu_isolated_6ch_track_multich|1.963|1.675|1.788|2|1.8564|
|et05_simu_isolated_6ch_track_multich|1.841|1.765|1.821|2.03|1.86422|

