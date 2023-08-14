# README
Paper: [EEND-SS: Joint End-to-End Neural Speaker Diarization and Speech Separation for Flexible Number of Speakers](https://arxiv.org/abs/2203.17068)

## Notes
- Results and the links to pre-trained models will be added later
- "concat_feats" in the training config file names refers to the model using the concatenation of TCN Bottleneck features and log-mel filterbank features (*[conf/tuning/train_diar_enh_convtasnet_concat_feats.yaml](conf/tuning/train_diar_enh_convtasnet_concat_feats.yaml) and [conf/tuning/train_diar_enh_convtasnet_concat_feats_adapt.yaml](conf/tuning/train_diar_enh_convtasnet_adapt.yaml)*). Only TCN Bottleneck features will be used by default.
- Model adaptation on 2 & 3 mixed speaker dataset can be conducted after pre-training on 2 speaker dataset. *Run [local/run_adapt.sh](local/run_adapt.sh) after running [run.sh](run.sh) with `num_spk=2`*.
- Post-processing can be applied to the separated audio signals using the speaker activity (diarization result) during inference. *To use this option, add `multiply_diar_result: True` in  [conf/tuning/decode_diar_enh.yaml](conf/tuning/decode_diar_enh.yaml) or [conf/tuning/decode_diar_enh_adapt.yaml](conf/tuning/decode_diar_enh_adapt.yaml)*.

## Results (to be added later)
### Environments

### Experimental Conditions
- sample_rate: 8k
- min_or_max: min
- threshold for DER calculation: 0.5
- collar torelance: 0.0 sec
- median filtering: 11 frames

### Libri2Mix (2 Speakers)
Model link: https://huggingface.co/soumi-maiti/libri2mix_eend_ss

### Libri3Mix (3 Speakers)
Model link: https://huggingface.co/soumi-maiti/libri3mix_eend_ss

### Libri2&3Mix (2 & 3 Speakers)
Model link: https://huggingface.co/soumi-maiti/libri23mix_eend_ss
