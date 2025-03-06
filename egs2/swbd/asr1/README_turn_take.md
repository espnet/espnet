# Turn taking prediction model

- config: [conf/train_asr_whisper_turn_taking.yaml](conf/train_asr_whisper_turn_taking.yaml)
- Model link: [https://huggingface.co/espnet/Turn_taking_prediction_SWBD](https://huggingface.co/espnet/Turn_taking_prediction_SWBD)

## ROC_AUC

|dataset|Continuation|Backchannel|Turn change|Interruption|Silence|Overall|
|---|---|---|---|---|---|---|
|decode_asr_chunk_asr_model_valid.loss.ave/test|93.3|89.4|90.8|91.3|95.1|92.0|
