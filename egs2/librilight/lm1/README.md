## Train speech Language Model

Train an LM using the prepared tokens and text data

Step 1:  Prepare discrete speech tokens for train/valid/test data and token vocabulary. This step is done externally.
Following files are needed: train: data/lm_train.txt, valid: data/lm_valid.txt test: data/lm_test.txt and token vocabulary: data/tokens.txt

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
