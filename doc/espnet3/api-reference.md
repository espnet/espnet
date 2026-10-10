---
title: ESPnet3 Python API Reference
---

# ESPnet3 Python API Reference

These pages are generated from the Sphinx-rendered docstrings in `espnet3/`.
Use the hand-written guides for concepts and workflows, then follow the links
below for exact class, function, argument, and return-value details.

## Pipeline and systems

- [`BaseSystem`](../guide/espnet3/systems/BaseSystem.html) — common stage API.
- [`ASRSystem`](../guide/espnet3/systems/ASRSystem.html) and
  [`TTSSystem`](../guide/espnet3/systems/TTSSystem.html) — task-specific systems.
- [`run_stages`](../guide/espnet3/utils/run_stages.html) and
  [`resolve_stages`](../guide/espnet3/utils/resolve_stages.html) — CLI dispatch.

## Components and data

- [`DataOrganizer`](../guide/espnet3/components/DataOrganizer.html),
  [`DatasetBuilder`](../guide/espnet3/components/DatasetBuilder.html), and
  [`DataLoaderBuilder`](../guide/espnet3/components/DataLoaderBuilder.html).
- [`ESPnetLightningModule`](../guide/espnet3/components/ESPnetLightningModule.html)
  and [`BaseMetric`](../guide/espnet3/components/BaseMetric.html).
- [`collect_stats`](../guide/espnet3/components/collect_stats.html) and
  [`instantiate_dataset_reference`](../guide/espnet3/components/instantiate_dataset_reference.html).

## Inference, parallelism, and publication

- [`InferenceProvider`](../guide/espnet3/systems/InferenceProvider.html) and
  [`InferenceRunner`](../guide/espnet3/systems/InferenceRunner.html).
- [`get_client`](../guide/espnet3/parallel/get_client.html) and
  [`set_parallel`](../guide/espnet3/parallel/set_parallel.html).
- [`InferenceModel`](../guide/espnet3/publication/InferenceModel.html),
  [`pack_model`](../guide/espnet3/utils/pack_model.html), and
  [`pack_demo`](../guide/espnet3/publication/pack_demo.html).

## Related guides

- [Stages](./stages/index.html) — stage-by-stage operational documentation.
- [Core packages](./core/index.html) — architecture and component guides.
- [Configuration files](./config/index.html) — recipe YAML reference.
- [Contributing](./contributing/index.html) — docstring and extension guidance.
