# TCPGen in RNN-T
# RESULTS
## Environments
- date: `Wed Jul  5 02:01:19 BST 2023`
- python version: `3.8.16 (default, Mar  2 2023, 03:21:46)  [GCC 11.2.0]`
- espnet version: `espnet 202304`
- pytorch version: `pytorch 2.0.1+cu117`
- Git hash: `6f33b9d9a999d4cd7e9bc0dcfc0ba342bdff7c17`
  - Commit date: `Thu Jun 29 02:16:09 2023 +0100`

- ASR Config: [conf/train_rnnt.yaml](conf/train_rnnt.yaml)
- Params: 27.13 M
- Model link: [https://huggingface.co/espnet/guangzhisun_librispeech100_asr_train_conformer_transducer_tcpgen500_deep_sche30_GCN6L_rep_suffix](https://huggingface.co/espnet/guangzhisun_librispeech100_asr_train_conformer_transducer_tcpgen500_deep_sche30_GCN6L_rep_suffix)

## exp/asr_train_conformer_transducer_tcpgen500_deep_sche30_GCN6L_rep_suffix
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.loss.ave/dev_clean|2703|54402|95.7|3.9|0.4|0.6|4.9|48.0|
|decode_asr_asr_model_valid.loss.ave/dev_other|2864|50948|85.8|12.6|1.6|1.9|16.1|77.0|
|decode_asr_asr_model_valid.loss.ave/test_clean|2620|52576|95.4|4.1|0.5|0.7|5.2|49.9|
|decode_asr_asr_model_valid.loss.ave/test_other|2939|52343|86.0|12.2|1.7|1.8|15.8|78.4|
|decode_b20_nolm_avebest/test_clean|2620|52576|0.0|0.0|100.0|0.0|100.0|100.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.loss.ave/dev_clean|2703|288456|98.4|1.0|0.7|0.6|2.3|48.0|
|decode_asr_asr_model_valid.loss.ave/dev_other|2864|265951|93.3|4.2|2.5|2.1|8.8|77.0|
|decode_asr_asr_model_valid.loss.ave/test_clean|2620|281530|98.3|1.0|0.7|0.6|2.3|49.9|
|decode_asr_asr_model_valid.loss.ave/test_other|2939|272758|93.6|3.8|2.6|1.9|8.3|78.4|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.loss.ave/dev_clean|2703|103998|95.3|3.5|1.2|0.6|5.3|48.0|
|decode_asr_asr_model_valid.loss.ave/dev_other|2864|95172|85.2|11.8|3.0|2.5|17.3|77.0|
|decode_asr_asr_model_valid.loss.ave/test_clean|2620|102045|95.3|3.4|1.3|0.6|5.4|49.9|
|decode_asr_asr_model_valid.loss.ave/test_other|2939|98108|85.5|11.0|3.5|2.2|16.7|78.4|


### Please cite our papers
```Bibtex
@INPROCEEDINGS{9687915,
  author={Sun, Guangzhi and Zhang, Chao and Woodland, Philip C.},
  booktitle={2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  title={Tree-Constrained Pointer Generator for End-to-End Contextual Speech Recognition},
  year={2021},
  volume={},
  number={},
  pages={780-787},
  doi={10.1109/ASRU51503.2021.9687915}
}

@inproceedings{Sun2022TreeconstrainedPG,
  title={Tree-constrained Pointer Generator with Graph Neural Network Encodings for Contextual Speech Recognition},
  author={Guangzhi Sun and C. Zhang and Philip C. Woodland},
  booktitle={Interspeech},
  year={2022}
}
```
