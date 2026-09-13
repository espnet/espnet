# mini_an4 BEATs pre-training smoke test

A deliberately tiny BEATs configuration that runs two BEATs iterations on the
Mini AN4 audio in CI. It exercises every stage of
`espnet3.systems.ssl.system.BeatsSystem`; the numbers are meaningless.

```bash
# Iteration 0: random-projection targets
python run.py --stages create_dataset infer collect_stats train \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# Iteration 1: tokenizer distilled from the iteration-0 encoder
python run.py --stages train_tokenizer infer train \
    --training_config conf/training_iter1.yaml \
    --train_tokenizer_config conf/training_tokenizer.yaml \
    --inference_config conf/inference.yaml
```
