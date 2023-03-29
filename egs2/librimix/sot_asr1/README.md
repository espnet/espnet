# Librimix SOT Recipe
This recipe implements the SOT ([Serialized Output Training](https://arxiv.org/pdf/2003.12687.pdf)) model for overlapped speech recognition on [LibriMix dataset](https://github.com/JorisCos/LibriMix).

Following are the details about this recipe.

## SOT (Serialized Output Training) Model
- The SOT model was proposed by Kanda et al. for end-to-end overlapped speech recognition.
- SOT model takes mixture speeches as inputs and re-formatted multi-speaker transcription as target labels.
- For the target label, a special symbol `<sc>` is used to represent the speaker change, and all texts from different speakers are concatenated by inserting `<sc>` between each other.
- During the training, the `<sc>` token will be treated as an user-defined unit
  - for word-based modeling, each word is separated by a space, and thus `<sc>` is automatically regarded as a single word unit;
  - for BPE-based modeling, add `--user_defined_symbols=<sc>` to `spm_train` and the `<sc>` token will not be mixed with other BPE subwords;
  - for char-based modeling, add `--add_nonsplit_symbol <sc>:2` for `text tokenizer` to maked sure that `<sc>` will not be split into `{"<", "s", "c", ">"}`.
- Take a 3-speaker mixture as an example, the multi-speaker transcription will be:
  - `Y = {text^1 <sc> text^2 <sc> text^3}`;
  - here, `text^n` refers to the transcription of `speaker n`;
  - the order of different texts is determined by their start times.
- More details about the SOT model could be found in https://arxiv.org/pdf/2003.12687.pdf

## Data Preparation
- We use Librimix data to verify the effectiveness of the SOT model.
- The Librimix data consists of 2- or 3-speaker mixtures simulated by combining the single-speaker speeches from [Librispeech](https://www.openslr.org/12) and noises from [WHAM!](https://wham.whisper.ai/) datasets.
- The official simulation process in https://github.com/JorisCos/LibriMix simulates fully-overlapped mixtures by default, which means speech from different speakers start at the same time.
- Following the setup in the SOT paper, we modify the original simulation code by `./local/create_librimix_from_metadata.patch` and provide modified `csv` files (in `./local/metadta_Libri2Mix_offset`) to generate mixtures with a random time delay from 1.0s to 1.5s.
- The format of our updated `csv` files are:
  - `mixture_ID,source_1_path,source_1_gain,source_2_path,source_2_gain,noise_path,noise_gain,offset`
  - compare with the original `csv` files in https://github.com/JorisCos/LibriMix, we add column `offset` to indicate the time delay during the simulation.
- **NOTE**: We can not directly comapre the results with `../asr1/RESULTS.md`, since results from `../asr1/RESULTS.md` are obtained on fully-overlapped Librimix data.

## RESULTS
### Environments
- date: `Thu Dec 29 13:36:46 CST 2022`
- python version: `3.8.13 (default, Mar 28 2022, 11:38:47)  [GCC 7.5.0]`
- espnet version: `espnet 202211`
- pytorch version: `pytorch 1.12.1`
- Git hash: ``
  - Commit date: ``
- ASR config: [conf/tuning/train_sot_asr_conformer_wavlm.yaml](conf/tuning/train_sot_asr_conformer_wavlm.yaml)
- Decode config: [conf/tuning/decode_sot.yaml](conf/tuning/decode_sot.yaml)
- Pretrained model: https://huggingface.co/espnet/pengcheng_librimix_asr_train_sot_asr_conformer_wavlm_raw_en_char_sp

### asr_train_sot_asr_conformer_wavlm_raw_en_char_sp
#### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_sot_asr_model_valid.acc.ave/dev|3000|123853|82.9|15.1|2.0|2.4|19.4|97.1|
|decode_sot_asr_model_valid.acc.ave/test|3000|111243|85.1|13.0|1.9|2.1|17.1|96.1|

#### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_sot_asr_model_valid.acc.ave/dev|3000|670222|92.2|4.9|2.9|2.7|10.6|97.1|
|decode_sot_asr_model_valid.acc.ave/test|3000|605408|93.2|4.1|2.6|2.3|9.1|96.1|


### Environments
- date: `Fri Jan 27 05:59:04 CST 2023`
- python version: `3.8.13 (default, Mar 28 2022, 11:38:47)  [GCC 7.5.0]`
- espnet version: `espnet 202211`
- pytorch version: `pytorch 1.12.1`
- Git hash: ``
  - Commit date: ``
- ASR config: [conf/tuning/train_sot_asr_conformer.yaml](conf/tuning/train_sot_asr_conformer.yaml)
- Decode config: [conf/tuning/decode_sot.yaml](conf/tuning/decode_sot.yaml)
- Pretrained model: https://huggingface.co/espnet/pengcheng_librimix_asr_train_sot_asr_conformer_raw_en_char_sp

### asr_train_sot_conformer_raw_en_char_sp
#### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_sot_asr_model_valid.acc.ave/dev|3000|123853|78.3|19.1|2.6|3.0|24.7|99.3|
|decode_sot_asr_model_valid.acc.ave/test|3000|111243|79.6|17.7|2.6|3.0|23.3|98.7|

#### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_sot_asr_model_valid.acc.ave/dev|3000|670222|90.1|6.3|3.6|3.5|13.4|99.3|
|decode_sot_asr_model_valid.acc.ave/test|3000|605408|90.7|5.7|3.6|3.3|12.6|98.7|
