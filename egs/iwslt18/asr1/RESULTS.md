# Transformer (BPE1k) + SpecAugment

| dataset                                                                                                                             | Snt  | Wrd   | Corr | Sub  | Del  | Ins | Err  | S.Err |
| ----------------------------------------------------------------------------------------------------------------------------------- | ---- | ----- | ---- | ---- | ---- | --- | ---- | ----- |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug/decode_dev.en_decode_pytorch_transformer_nolm     | 2084 | 31492 | 90.4 | 7.4  | 2.3  | 7.0 | 16.6 | 62.0  |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug/decode_test.en_decode_pytorch_transformer_nolm    | 2092 | 32546 | 90.1 | 7.7  | 2.2  | 5.5 | 15.3 | 64.7  |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug/decode_dev2010.en_decode_pytorch_transformer_nolm | 888  | 17708 | 81.8 | 8.3  | 9.9  | 4.4 | 22.6 | 81.1  |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug/decode_tst2010.en_decode_pytorch_transformer_nolm | 1568 | 27605 | 80.9 | 6.5  | 12.6 | 3.6 | 22.7 | 79.4  |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug/decode_tst2013.en_decode_pytorch_transformer_nolm | 993  | 18160 | 76.4 | 11.8 | 11.7 | 4.0 | 27.6 | 85.2  |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug/decode_tst2014.en_decode_pytorch_transformer_nolm | 1305 | 21685 | 74.6 | 8.5  | 16.9 | 2.8 | 28.3 | 83.7  |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug/decode_tst2015.en_decode_pytorch_transformer_nolm | 1080 | 18288 | 69.5 | 12.4 | 18.2 | 9.8 | 40.3 | 84.6  |

# RNN (BPE1k) decoding

| dataset                                                                                | Snt  | Wrd   | Corr | Sub  | Del  | Ins  | Err  | S.Err |
| -------------------------------------------------------------------------------------- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ----- |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train_rnn_spm_bpe1000/decode_dev.en_decode     | 2084 | 31492 | 86.8 | 10.0 | 3.3  | 9.0  | 22.3 | 71.0  |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train_rnn_spm_bpe1000/decode_test.en_decode    | 2092 | 60585 | 88.7 | 7.5  | 3.8  | 5.3  | 16.6 | 71.4  |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train_rnn_spm_bpe1000/decode_dev2010.en_decode | 888  | 17708 | 78.6 | 9.8  | 11.6 | 3.9  | 25.3 | 84.5  |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train_rnn_spm_bpe1000/decode_tst2010.en_decode | 1568 | 27605 | 75.9 | 8.2  | 15.9 | 2.7  | 26.8 | 84.4  |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train_rnn_spm_bpe1000/decode_tst2013.en_decode | 993  | 18160 | 71.1 | 14.1 | 14.9 | 3.6  | 32.5 | 88.2  |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train_rnn_spm_bpe1000/decode_tst2014.en_decode | 1305 | 21685 | 73.0 | 10.8 | 16.2 | 3.0  | 30.0 | 84.4  |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train_rnn_spm_bpe1000/decode_tst2015.en_decode | 1080 | 18288 | 71.6 | 13.1 | 15.3 | 11.7 | 40.1 | 85.0  |

# RNN (character unit) + RNNLM

| dataset                                                                | Snt  | Wrd   | Corr | Sub  | Del  | Ins | Err  | S.Err |
| ---------------------------------------------------------------------- | ---- | ----- | ---- | ---- | ---- | --- | ---- | ----- |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train/decode_dev.en_decode     | 2084 | 31479 | 85.8 | 10.7 | 3.5  | 8.3 | 22.5 | 71.8  |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train/decode_test.en_decode    | 2092 | 32538 | 86.0 | 10.4 | 3.5  | 6.3 | 20.3 | 71.9  |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train/decode_dev2010.en_decode | 888  | 17708 | 75.2 | 10.9 | 13.9 | 3.4 | 28.2 | 86.1  |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train/decode_tst2010.en_decode | 1568 | 27605 | 72.0 | 8.8  | 19.2 | 2.3 | 30.3 | 86.9  |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train/decode_tst2013.en_decode | 993  | 18160 | 66.6 | 14.6 | 18.7 | 3.2 | 36.6 | 90.1  |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train/decode_tst2014.en_decode | 1305 | 21685 | 68.7 | 10.9 | 20.3 | 2.5 | 33.8 | 86.6  |
| exp/train_nodevtest_sp.en_lc.rm_pytorch_train/decode_tst2015.en_decode | 1080 | 18288 | 62.2 | 14.4 | 23.4 | 7.8 | 45.6 | 87.6  |
