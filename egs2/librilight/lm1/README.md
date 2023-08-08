## Train speech Language Model

Train an LM using the prepared tokens and text data

Step 1:  Prepare discrete speech tokens for train/valid/test data and token vocabulary. This step is done externally.

### External Hubert feature extraction
Discrete speech tokens extracted using `fairseq` library. Example:  Hubert speech tokens can be extracted using: https://github.com/facebookresearch/fairseq/tree/main/examples/hubert/simple_kmeans and using script ```local/extract_feat.sh```.

Following prepared files are needed for this recipe:
* **train: _data/lm_train.txt_**
* **valid: _data/lm_valid.txt_**
* **test: _data/lm_test.txt_**
* **token vocabulary: _data/tokens.txt_**


`token vocabulary` can be generated using `local/create_token_voc.py`

Discrete speech tokens contains lines with format:  <utt_id> <space_separated_discrete_speech_tokens> . Example:
```
100_2315_04_baum_sea_fairies_64kb_0000	20 18 48 47 34 11 14 21 34 36 14 21 35 41 19 45 4 7 44 37 17 7 30 0 8 3 44 11 36 25 47 17 24 44 49 32 29 24 26 16 12 46 9 20 33 20 33 20 18 6 8 42 41 19 45 7 37 17 24 17 30 0 8 3 44 11
```
Step 2: Move preprocessed data for LM training
```
./local/data_prep.sh
```

Step 3: Train LM
```
./run.sh
```

## Speech Language Model with LSTM

### Environments
* date: `Oct 16 11:22:52 CDT 2022`
* pytorch version: `pytorch 1.12.1`
* Vocabulary: 50
* LM config: https://github.com/soumimaiti/espnet/blob/train_speechlm/egs2/librilight/lm1/conf/tuning/train_lm_rnn_unit1024_nlayers3_dropout0.2_epoch30.yaml
* Pretrained model: https://huggingface.co/soumi-maiti/speech-ulm-lstm/tree/main
* Perplexity on Librispeech test (test-clean+test-other): 3.29
