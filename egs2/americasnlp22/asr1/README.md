This is the baseline setup for the ASR task of
[the second AmericasNLP competition](http://turing.iimas.unam.mx/americasnlp/st.html).

## XLS-R frontend and CTC mode:

- ASR config: [conf/train_asr_transformer.yaml](conf/train_asr_transformer.yaml)
- Pretrained models on Hugging Face Hub:
  - Bribri: https://huggingface.co/espnet/americasnlp22-asr-bzd
  - Guarani: https://huggingface.co/espnet/americasnlp22-asr-gug
  - Kotiria: https://huggingface.co/espnet/americasnlp22-asr-gvc
  - Quechua: https://huggingface.co/espnet/americasnlp22-asr-qwe
  - Wa'ikhana: https://huggingface.co/espnet/americasnlp22-asr-tav

## Results:
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.cer_ctc.best/dev_bzd|250|2056|15.3|65.1|19.6|7.5|92.3|100.0|
|decode_asr_asr_model_valid.cer_ctc.best/dev_gug|93|391|11.5|73.7|14.8|12.5|101.0|100.0|
|decode_asr_asr_model_valid.cer_ctc.best/dev_gvc|253|2206|12.4|72.4|15.1|6.7|94.2|99.6|
|decode_asr_asr_model_valid.cer_ctc.best/dev_qwe|250|11465|18.7|67.0|14.3|4.3|85.6|100.0|
|decode_asr_asr_model_valid.cer_ctc.best/dev_tav|250|1201|3.0|83.1|13.9|17.0|114.0|99.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.cer_ctc.best/dev_bzd|250|10083|64.0|15.1|20.9|9.2|45.2|100.0|
|decode_asr_asr_model_valid.cer_ctc.best/dev_gug|93|2946|83.4|7.9|8.7|8.7|25.3|100.0|
|decode_asr_asr_model_valid.cer_ctc.best/dev_gvc|253|13453|64.7|15.5|19.9|10.2|45.6|99.6|
|decode_asr_asr_model_valid.cer_ctc.best/dev_qwe|250|95334|78.6|8.0|13.4|10.1|31.5|100.0|
|decode_asr_asr_model_valid.cer_ctc.best/dev_tav|250|8606|57.5|19.9|22.7|12.0|54.5|99.6|

