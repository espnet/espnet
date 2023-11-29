# Speech Enhancement Frontend Recipe

This is the common recipe for ESPnet2 joint-task with speech enhancement frontend. Following are the directory structure of speech enhancement and joint-task recipes:

```
egs2/
├── chime4/
│   ├── enh1/
│   ├── enh_asr1/
│   └── asr1/
├── l3das22/
│   └── enh1/
|   │   ├── conf/
|   │   ├── local/
|   |   │   ├── data.sh
|   |   │   ├── metric.sh
│   |   │   └── ...
|   │   ├── enh.sh -> ../../TEMPLATE/enh1/enh.sh
|   │   ├── run.sh
|   │   └── ...
├── lt_slurp_spatialized/
│   └── enh1/
├── slurp_spatialized/
│   ├── enh_asr1/
|   │   ├── enh_asr.sh -> ../../TEMPLATE/enh_asr1/enh_asr.sh
|   │   ├── run.sh
|   │   └── ...
│   └── asr1/
├── ...
└── TEMPLATE/
    ├── enh1/
    │   └── enh.sh
    ├── enh_asr1/
    │   └── enh_asr.sh
    ├── enh_diar1/
    │   └── enh_diar.sh
    ├── enh_st1/
    │   └── enh_st.sh
    └── ...
```

## Introduction to enh_asr.sh
In `egs2/TEMPLATE/enh_asr1/enh_asr.sh`, 17 stages are included. Most of the stages are similar to [asr.sh](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/asr.sh) and [enh.sh](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/enh1/enh.sh).

#### stage 1 to stage 5: data preparation stages
- Stage 1: Data preparation
- Stage 2: Speech perturbation
- Stage 3: Format wav.scp
- Stage 4: Remove short data
- Stage 5: Generate token_list using BPE
#### stage 6 to stage 9: language model training steps
- Stage 6: LM collect stats
- Stage 7: LM Training
- Stage 8: Calc perplexity
- Stage 9: Ngram Training

#### stage 10 to stage 11: joint-task training steps
- Stage 10: Collect stats for the joint task.
- Stage 11: Joint task Training

We have created `EnhS2TTask` in `espnet2/tasks/enh_s2t_train.py`, which is used to train the `ESPnetEnhS2TModel(AbsESPnetModel)` defined in `espnet2/enh/espnet_enh_s2t_model.py`. The ESPnetEnhS2TModel takes a front-end enh_model, and a back-end s2t_model (such as ASR, SLU, ST, and SD models) as inputs to build a joint-model.

Related python files:
```
espnet2/
├── bin/
│   └── enh_s2t_train.py
├── enh/
│   └── espnet_enh_s2t_model.py
├── tasks/
│   └── enh_s2t.py
└── ...
```

#### stage 12 to stage 13: Inference stages: Decoding and enhancing
- Stage 12: downstream tasks (ASR) Decoding
- Stage 13: Enhance Speech

Related python files:
```
espnet2/
├── bin/
│   ├── asr_inference.py
│   ├── diar_inference.py
│   ├── enh_inference.py
│   └── st_inference.py
└── ...
```


#### stage 14 to stage 15: Scoring recognition and SSE results
- Stage 14: Scoring ASR
- Stage 15: Scoring Enhancement


#### stage 16 to stage 17: model uploading steps
- Stage 16: Pack model
- Stage 17: Upload model to Hugging Face
