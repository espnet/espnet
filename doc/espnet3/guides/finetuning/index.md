# Finetuning

Most ESPnet3 finetuning work reduces to three questions:

1. Which dataset do you train on?
2. Which model do you actually fine-tune?
3. How much of the old model do you keep or replace?

Use this section for those three topics.

<DocCards :cols="2">
  <DocCard
    title="Custom dataset"
    desc="Point training and inference at your own dataset module and manifests."
    icon="tabler:database"
    href="./custom-dataset.html"
  />
  <DocCard
    title="Customize the model"
    desc="Use `src/model.py`, keep or replace old weights, and switch from the task bridge when needed."
    icon="tabler:puzzle"
    href="./custom-model.html"
  />
  <DocCard
    title="Adding a stage"
    desc="Use the contributor guide when fine-tuning needs extra preparation or export stages."
    icon="tabler:route"
    href="../../contributing/adding-a-stage.html"
  />
  <DocCard
    title="Customize the training loop"
    desc="Use recipe-local LightningModule, trainer, or System code when the default loop is not enough."
    icon="tabler:settings-cog"
    href="./training-loop.html"
  />
</DocCards>
