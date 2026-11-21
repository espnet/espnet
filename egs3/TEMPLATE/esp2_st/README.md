# ESPnet3 speech translation template

This template uses `espnet3.systems.esp2_st.ESP2STSystem` to train and evaluate an ESPnet2 speech translation task. It provides separate source and target tokenizer preparation where configured, along with statistics, training, inference, measurement, and publication stages. The recipe provides corpus-specific dataset and configuration code.

## Data

Each example contains speech and the target translation. A source transcript (`src_text`) can also be supplied when the training setup uses an auxiliary source-text objective.

```yaml
fields:
  speech: audio
  src_text: text
  text: text
```

The inference contract accepts speech and produces translated text. The default contract selects the first channel and requires 16 kHz audio; recipes can override it when needed.

## Metrics

The system provides BLEU, chrF2, and TER through its BLEU metric implementation. Bind the reference to the dataset's target text and the hypothesis to the generated text.

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.esp2_st.metrics.bleu.BLEU
    inputs:
      ref: dataset:text
      hyp: text
```

## Typical workflow

Run from the corpus recipe directory after configuring language pair, data source, and model. The recipe's `run.py` calls `launch()` to combine template defaults with its own configuration.

```bash
python run.py --stages create_dataset train_tokenizer collect_stats train \
  --training_config conf/training.yaml
python run.py --stages infer \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml
python run.py --stages measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

See [mini_an4](../../mini_an4/esp2_st) for a small runnable example and [MuST-C](../../must_c/esp2_st) for a corpus recipe.
