## Train speech Language Model

Train an LM using the prepared tokens and text data:

Step 1:  Prepare discrete speech tokens for train/valid/test data and token vocabulary. Example: train: data/lm_train.txt, valid: data/lm_valid.txt test: data/lm_test.txt and token vocabulary: data/tokens.txt

Step 2: Move preprocessed data for LM training
```
mkdir -p data/token_list/word
cp data/tokens.txt data/token_list/word/

lm_train_text=data/lm_train.txt
data_feats=dump/raw
mkdir -p ${data_feats}
cat ${lm_train_text} | awk ' { if( NF != 1 ) print $0; } ' > "${data_feats}/lm_train.txt"
```

Step 3: Train model
```
./run.sh
```