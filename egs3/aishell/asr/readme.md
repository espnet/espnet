# AISHELL-1 ASR recipe

Place the OpenSLR 33 `data_aishell` directory under `download/`, or set
`AISHELL` to either that directory or its parent, before running
`create_dataset`/`train`. The expected corpus layout is:

```
data_aishell/
├── transcript/aishell_transcript_v0.8.txt
└── wav/{train,dev,test}/
```

## Quick start

```bash
# 1) Validate the corpus layout, then train the default E-Branchformer model
python run.py --stages create_dataset \
    --training_config conf/tuning/training_e_branchformer.yaml

python run.py --stages train \
    --training_config conf/tuning/training_e_branchformer.yaml

# 2) Decode
python run.py --stages infer \
    --training_config conf/tuning/training_e_branchformer.yaml \
    --inference_config conf/inference.yaml

# 3) Score
python run.py --stages measure \
    --training_config conf/tuning/training_e_branchformer.yaml \
    --metrics_config conf/metrics.yaml
```

The recipe uses a character tokenizer and reports character error rate (CER).
