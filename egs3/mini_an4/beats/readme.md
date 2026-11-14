# mini_an4 BEATs pre-training smoke test

A deliberately tiny BEATs configuration that runs two BEATs iterations on the
Mini AN4 audio in CI. It exercises every stage of
`espnet3.systems.beats.system.BeatsSystem`; the numbers are meaningless.

```bash
# Iteration 0: random-projection targets -> encoder, codebook usage, bundle
python run.py --stages pretrain measure pack_model \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml \
    --publication_config conf/publication.yaml

# Iteration 1: tokenizer distilled from the iteration-0 encoder -> encoder
python run.py --stages pretrain measure \
    --training_config conf/training_iter1.yaml \
    --train_tokenizer_config conf/training_tokenizer.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```
