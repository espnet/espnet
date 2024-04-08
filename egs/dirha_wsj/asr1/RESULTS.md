# RNN results
## Mic: Beam_Circular_Array
### (pytorch) 2-layer vggblstmp, add attention, batchsize 15;
#### CER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_Beam_Circular_Array_pytorch_train_no_preprocess/decode_dirha_real_Beam_Circular_Array_decode_lm_word65000/result.txt:| 409 | 39842 | 82.5 | 9.3 | 8.2 | 3.8 | 21.3 | 83.4 |
#### WER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_Beam_Circular_Array_pytorch_train_no_preprocess/decode_dirha_real_Beam_Circular_Array_decode_lm_word65000/result.wrd.txt:| 409 | 6762 | 69.3 | 25.6 | 5.0 | 4.3 | 35.0 | 83.4 |

### add SpecAug;
#### CER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_Beam_Circular_Array_pytorch_train_specaug/decode_dirha_real_Beam_Circular_Array_decode_lm_word65000/result.txt:| 409 | 39842 | 83.9 | 7.2 | 8.9 | 2.8 | 19.0 | 80.9 |
#### WER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_Beam_Circular_Array_pytorch_train_specaug/decode_dirha_real_Beam_Circular_Array_decode_lm_word65000/result.wrd.txt:| 409 | 6762 | 72.6 | 20.6 | 6.8 | 2.3 | 29.7 | 80.9 |

## Mic: Beam_Linear_Array
### (pytorch) 2-layer vggblstmp, add attention, batchsize 15;
#### CER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_Beam_Linear_Array_pytorch_train_no_preprocess/decode_dirha_real_Beam_Linear_Array_decode_lm_word65000/result.txt:| 409 | 39842 | 83.7 | 9.9 | 6.4 | 5.2 | 21.6 | 85.6 |
#### WER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_Beam_Linear_Array_pytorch_train_no_preprocess/decode_dirha_real_Beam_Linear_Array_decode_lm_word65000/result.wrd.txt:| 409 | 6762 | 69.1 | 27.6 | 3.3 | 5.9 | 36.8 | 85.6 |

### add SpecAug;
#### CER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_Beam_Linear_Array_pytorch_train_specaug/decode_dirha_real_Beam_Linear_Array_decode_lm_word65000/result.txt:| 409 | 39842 | 87.2 | 6.9 | 5.9 | 3.3 | 16.2 | 76.5 |
#### WER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_Beam_Linear_Array_pytorch_train_specaug/decode_dirha_real_Beam_Linear_Array_decode_lm_word65000/result.wrd.txt:| 409 | 6762 | 76.2 | 19.7 | 4.1 | 3.2 | 27.0 | 76.5 |

## Mic: L1C
### (pytorch) 2-layer vggblstmp, add attention, batchsize 15;
#### CER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_L1C_pytorch_train_no_preprocess/decode_dirha_real_L1C_decode_lm_word65000/result.txt:| 409 | 39842 | 84.1 | 11.3 | 4.6 | 8.6 | 24.6 | 83.6 |
#### WER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_L1C_pytorch_train_no_preprocess/decode_dirha_real_L1C_decode_lm_word65000/result.wrd.txt:| 409 | 6762 | 69.3 | 28.5 | 2.2 | 9.3 | 39.9 | 83.6 |

### add SpecAug;
#### CER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_L1C_pytorch_train_specaug/decode_dirha_real_L1C_decode_lm_word65000/result.txt:| 409 | 39842 | 86.2 | 9.3 | 4.5 | 6.4 | 20.2 | 81.2 |
#### WER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_L1C_pytorch_train_specaug/decode_dirha_real_L1C_decode_lm_word65000/result.wrd.txt:| 409 | 6762 | 73.7 | 23.2 | 3.1 | 5.3 | 31.6 | 81.2 |
