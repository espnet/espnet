# Customize the model

If you want to change the model more seriously, the practical ESPnet3 pattern
is usually:

1. add `src/model.py`
2. instantiate that model from `training.yaml`
3. keep dataset and inference code aligned with the new model contract

For many recipes, that is all you need to know first.

::: important
If the old task-side model path is too restrictive, put your model in
`egs3/<recipe>/<system>/src/model.py` and instantiate it directly from
`training.yaml`.
:::

That is the clean ESPnet3 answer to "I want my own model now."

## Typical recipe-local layout

```text
egs3/<recipe>/<system>/
  conf/
    training.yaml
    inference.yaml
  src/
    model.py
    inference.py
  dataset/
    dataset.py
```

The important point is that `src/` is where recipe-local Python code lives.

Typical responsibilities:

- `src/model.py`: model class
- `src/inference.py`: output formatting or inference helpers

## The main config change

The key change is in `training.yaml`.

### Old task-bridge style

```yaml
task: espnet2.tasks.asr.ASRTask

model:
  encoder: transformer
  decoder: transformer
  model_conf:
    ctc_weight: 0.3
```

### Direct ESPnet3 recipe-local model

```yaml
task:

model:
  _target_: src.model.MyModel
  hidden_size: 256
  vocab_size: 500
```

So the practical migration is:

- leave `task` unset
- make `model` a direct Hydra instantiation block

That is the main switch from system-driven to recipe-local model ownership.

## A minimal src/model.py

At minimum, this can just be a normal PyTorch module.

```python
import torch


class MyModel(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.encoder = torch.nn.Linear(80, hidden_size)
        self.head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, speech):
        hidden = self.encoder(speech)
        return self.head(hidden)
```


## Inference alignment

Training is not the only place that changes.

If the model output no longer matches the old inference assumptions, update the
recipe-local inference helper too.

Typical location:

```text
src/inference.py
```

For example, if the model returns a custom object or a different hypothesis
shape, `output_fn` should convert it into the final dict written by inference.

Conceptually:

```python
def build_output(data, model_output, idx):
    return {
        "utt_id": data["utt_id"],
        "hyp": ...,
        "ref": data.get("text", ""),
    }
```

Then `inference.yaml` points to it:

```yaml
output_fn: src.inference.build_output
```

## Reusing an existing checkpoint

This is where the old "load checkpoint" idea usually ends up.

If the recipe-local model still matches an older checkpoint closely enough, you
can reuse that checkpoint here.

Typical cases are:

- full reuse when the architecture still matches
- partial reuse when only the backbone matches
- keep the backbone and replace the head
- use a PEFT-style wrapper model that owns the adaptation logic itself

So in practice, checkpoint reuse is usually part of custom-model work, not a
separate topic.

## A useful rule of thumb

Prefer the espnet2 task path when:

- the model is still basically the same
- only small config changes are needed

Prefer `src/model.py` when:

- the model itself is now recipe-local logic
- the old task abstraction is in the way
- future work will keep changing the architecture

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Custom dataset"
    desc="Make sure the dataset contract matches the recipe-local model."
    icon="tabler:database"
    href="./custom-dataset.html"
  />
  <DocCard
    title="Data pipeline"
    desc="See how dataset outputs and collate behavior affect custom models."
    icon="tabler:stack-2"
    href="../migrating-from-espnet2/data-pipeline.html"
  />
  <DocCard
    title="Training Config"
    desc="See where the recipe-local model is instantiated and optimized."
    icon="tabler:settings-2"
    href="../../config/train_config.html"
  />
  <DocCard
    title="Inference Config"
    desc="See how the custom model connects to provider, runner, and output_fn."
    icon="tabler:wave-sine"
    href="../../config/infer_config.html"
  />
</DocCards>
