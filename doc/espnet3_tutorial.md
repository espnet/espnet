# ESPnet3 Tutorial

ESPnet3 recipes use a Python entry point, YAML configuration roles and a system's
`StageSpec` declarations. The working ASR example is
`egs3/mini_an4/esp2_asr`; defaults live under `egs3/TEMPLATE/esp2_asr/conf/`.

```python
from espnet3.systems.esp2_asr.system import ASRSystem
from espnet3.systems.launch import launch

if __name__ == "__main__":
    launch(ASRSystem)
```

From the Mini-AN4 recipe directory, select the required stages explicitly:

```bash
python run.py \
  --stages create_dataset train_tokenizer collect_stats train infer measure \
  --training_config conf/training_asr_transformer.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

Configuration roles are `training`, `inference`, `metrics`, `publication` and
`demo` for the standard ASR system. A `--<role>_config` is loaded only when passed;
that role's packaged template defaults are merged with the recipe override.
The launcher resolves stage configs, then executes stages in their declared order.
It does not add prerequisite stages automatically. In this version, selected ASR
`train` needs `train_tokenizer` selected too, and `measure` needs `infer` selected
too; saved config recovery with `--exp_dir` does not create omitted stage entries.

The provider constructs datasets/models, and `ASRInferenceRunner` adapts native
ESPnet2 n-best output to `text`. `contract.inputs` / `contract.outputs` in the
inference YAML are checked through the same runner path used by recipe inference,
`InferenceModel` and the default demo. Metrics declare their own contracts and
bind inputs separately in `metrics.yaml`.

See [Adding a stage](./espnet3/contributing/adding-a-stage.html) for a complete
custom stage and contract example, [Developer setup](./espnet3/contributing/dev-setup.html)
for dependencies, and [CI and PR](./espnet3/contributing/ci-and-pr.html) for checks.
