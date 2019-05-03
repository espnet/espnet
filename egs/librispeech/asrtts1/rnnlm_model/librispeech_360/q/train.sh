#!/bin/bash
cd /export/a08/obask/espnet_cyc/egs/librispeech/asr1
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
lm_train.py --ngpu 1 --backend pytorch --verbose 1 --outdir exp/train_rnnlm_pytorch_1layer_unit1000_sgd_bs300_unigram5000 --train-label data/local/lm_train/train.txt --valid-label data/local/lm_train/valid.txt --resume --layer 1 --unit 1000 --opt sgd --batchsize 300 --epoch 20 --maxlen 40 --dict data/lang_char/train_460_units.txt 
EOF
) >exp/train_rnnlm_pytorch_1layer_unit1000_sgd_bs300_unigram5000/train.log
time1=`date +"%s"`
 ( lm_train.py --ngpu 1 --backend pytorch --verbose 1 --outdir exp/train_rnnlm_pytorch_1layer_unit1000_sgd_bs300_unigram5000 --train-label data/local/lm_train/train.txt --valid-label data/local/lm_train/valid.txt --resume --layer 1 --unit 1000 --opt sgd --batchsize 300 --epoch 20 --maxlen 40 --dict data/lang_char/train_460_units.txt  ) 2>>exp/train_rnnlm_pytorch_1layer_unit1000_sgd_bs300_unigram5000/train.log >>exp/train_rnnlm_pytorch_1layer_unit1000_sgd_bs300_unigram5000/train.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/train_rnnlm_pytorch_1layer_unit1000_sgd_bs300_unigram5000/train.log
echo '#' Finished at `date` with status $ret >>exp/train_rnnlm_pytorch_1layer_unit1000_sgd_bs300_unigram5000/train.log
[ $ret -eq 137 ] && exit 100;
touch exp/train_rnnlm_pytorch_1layer_unit1000_sgd_bs300_unigram5000/q/sync/done.6542
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o exp/train_rnnlm_pytorch_1layer_unit1000_sgd_bs300_unigram5000/q/train.log -l 'hostname=b1[12345678]*|c*,gpu=1' -q g.q -l mem_free=2G,ram_free=2G   /export/a08/obask/espnet_cyc/egs/librispeech/asr1/exp/train_rnnlm_pytorch_1layer_unit1000_sgd_bs300_unigram5000/q/train.sh >>exp/train_rnnlm_pytorch_1layer_unit1000_sgd_bs300_unigram5000/q/train.log 2>&1
