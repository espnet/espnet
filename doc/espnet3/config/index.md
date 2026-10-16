# Config files

Every ESPnet3 recipe is driven by a small set of YAML files under `conf/`. Each
stage reads exactly one of them; there are no CLI key-value overrides.

<DocCards :cols="2">
  <DocCard
    title="training.yaml"
    desc="Dataset, model, trainer, optimizer, and dataloader settings for create_dataset, collect_stats, and train."
    icon="tabler:school"
    href="./train_config.html"
  />
  <DocCard
    title="inference.yaml"
    desc="Decoder settings, test sets, and output layout for the infer stage."
    icon="tabler:bolt"
    href="./infer_config.html"
  />
  <DocCard
    title="publication.yaml"
    desc="Model card and Hugging Face Hub settings for pack_model / upload_model."
    icon="tabler:package-export"
    href="./publish_config.html"
  />
  <DocCard
    title="demo.yaml"
    desc="Gradio demo definition for pack_demo / upload_demo."
    icon="tabler:device-desktop"
    href="./demo_config.html"
  />
</DocCards>

## Related pages

- [Stages](../stages/index.html) — which stage reads which config file.
- [metrics.yaml](../stages/metrics.html) — documented together with the `measure` stage.
- [Experiment naming](../experiment_naming_examples.html) — how `exp_tag` and `exp_dir` are derived.
