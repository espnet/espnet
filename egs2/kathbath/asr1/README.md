# E-Branchformer
- ASR config: conf/train_asr.yaml
- Params: 35M
- Size: 140.03 MB
- Training time: 16349 to 34072 seconds
- GPU: a single 3090 24GB

## Environments
- date: `Thu Jul 27 01:21:08 IST 2023`
- python version: `3.8.10 (default, May 26 2023, 14:05:08)  [GCC 9.4.0]`
- espnet version: `espnet 202304`
- pytorch version: `pytorch 1.10.1+cu113`
- Git hash: `4c8aeda5de44f08d3617cdfc7aeb30bdf3d53d72`
  - Commit date: `Tue Jul 25 19:35:21 2023 +0530`

### WER

|lang|exp|test|test_known|test_known_noisy|test_noisy|
|---|---|---|---|---|---|
|urdu|asr_train_asr_raw_urdu_bpe500|14.8|13.1|15.8|21.1|
|telugu|asr_train_asr_raw_telugu_bpe500|25.1|21.2|23.9|28.4|
|kannada|asr_train_asr_raw_kannada_bpe500|19.2|17.9|18.7|27.4|
|malayalam|asr_train_asr_raw_malayalam_bpe500|39.2|30.0|31.3|43.1|
|bengali|asr_train_asr_raw_bengali_char|17.2|13.2|16.2|21.8|
|odia|asr_train_asr_raw_odia_bpe500|23.6|16.7|19.9|27.5|
|gujarati|asr_train_asr_raw_gujarati_char|19.3|15.1|18.4|27.2|
|hindi|asr_train_asr_raw_hindi_bpe500|12.5|10.1|12.6|14.3|
|tamil|asr_train_asr_raw_tamil_bpe500|23.3|20.1|24.3|24.2|
|punjabi|asr_train_asr_raw_punjabi_bpe500|15.9|14.3|14.2|24.9|
|sanskrit|asr_train_asr_raw_sanskrit_bpe200|40.3|27.7|39.3|49.8|
|marathi|asr_train_asr_raw_marathi_bpe500|16.6|15.1|16.7|19.7|

### CER

|lang|exp|test|test_known|test_known_noisy|test_noisy|
|---|---|---|---|---|---|
|urdu|asr_train_asr_raw_urdu_bpe500|4.8|3.7|4.7|7.7|
|telugu|asr_train_asr_raw_telugu_bpe500|5.0|3.7|4.6|7.3|
|kannada|asr_train_asr_raw_kannada_bpe500|4.0|3.3|3.9|7.1|
|malayalam|asr_train_asr_raw_malayalam_bpe500|7.8|5.4|6.2|9.4|
|bengali|asr_train_asr_raw_bengali_char|3.9|2.6|3.7|5.6|
|odia|asr_train_asr_raw_odia_bpe500|6.0|3.3|4.4|8.1|
|gujarati|asr_train_asr_raw_gujarati_char|5.2|3.5|4.9|9.1|
|hindi|asr_train_asr_raw_hindi_bpe500|4.2|3.1|4.3|5.4|
|tamil|asr_train_asr_raw_tamil_bpe500|4.3|3.3|4.9|4.7|
|punjabi|asr_train_asr_raw_punjabi_bpe500|5.2|3.8|4.5|9.4|
|sanskrit|asr_train_asr_raw_sanskrit_bpe200|10.1|5.6|9.8|14.6|
|marathi|asr_train_asr_raw_marathi_bpe500|4.1|3.8|4.3|5.2|


### Pretrained models

|lang|hugging_face link|
|---|---|
|urdu|https://huggingface.co/viks66/asr_train_asr_raw_urdu_bpe500|
|telugu|https://huggingface.co/viks66/asr_train_asr_raw_telugu_bpe500|
|kannada|https://huggingface.co/viks66/asr_train_asr_raw_kannada_bpe500|
|malayalam|https://huggingface.co/viks66/asr_train_asr_raw_malayalam_bpe500|
|bengali|https://huggingface.co/viks66/asr_train_asr_raw_bengali_char|
|odia|https://huggingface.co/viks66/asr_train_asr_raw_odia_bpe500|
|gujarati|https://huggingface.co/viks66/asr_train_asr_raw_gujarati_char|
|hindi|https://huggingface.co/viks66/asr_train_asr_raw_hindi_bpe500|
|tamil|https://huggingface.co/viks66/asr_train_asr_raw_tamil_bpe500|
|punjabi|https://huggingface.co/viks66/asr_train_asr_raw_punjabi_bpe500|
|sanskrit|https://huggingface.co/viks66/asr_train_asr_raw_sanskrit_bpe500|
|marathi|https://huggingface.co/viks66/asr_train_asr_raw_marathi_bpe500|
