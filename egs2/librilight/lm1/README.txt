## Train speech Language Model

Train an LM using the prepared tokens and text data:

Step 1: Prepare data/tokens and move them to data

```
mkdir -p data/token_list/word
cp data/tokens.txt data/token_list/word/

lm_train_text=data/lm_train.txt
data_feats=dump/raw
mkdir -p ${data_feats}
cat ${lm_train_text} | awk ' { if( NF != 1 ) print $0; } ' > "${data_feats}/lm_train.txt"
```

Step 2: Train model
```
./run.sh
```