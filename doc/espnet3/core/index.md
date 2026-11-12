---
title: ESPnet3 Core Packages
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Core Packages

This section is a hub for the main Python packages under `espnet3/`.


## Main packages

- Systems: `espnet3/systems/`
- Components: `espnet3/components/`
- Config: recipe and stage YAML files
- Parallel: `espnet3/parallel/`
- Demo: `espnet3/publication/demo/`
- Utilities: `espnet3/utils/`

## Start here

<DocCards :cols="3">
  <DocCard
    title="System and stages"
    desc="Read how run.py, System classes, stages, config slots, and stage logs fit together."
    icon="tabler:hierarchy-2"
    href="../stages/index.html"
  />
  <DocCard
    title="Components"
    desc="See the reusable data, modeling, trainer, and metric layers."
    icon="tabler:puzzle"
    href="./components/index.html"
  />
  <DocCard
    title="Datasets"
    desc="Start from the dataset internals hub for references, builders, organizers, and dataloaders."
    icon="tabler:database"
    href="./components/data-organizer.html"
  />
  <DocCard
    title="Config"
    desc="See training, inference, metrics, publication, demo, and parallel config files."
    icon="tabler:settings-2"
    href="../config/index.html"
  />
  <DocCard
    title="Parallel"
    desc="Read the provider and runner layer for distributed execution."
    icon="tabler:stack-2"
    href="./parallel/index.html"
  />
  <DocCard
    title="Demo"
    desc="See the packaged demo runtime, UI definition, and pack pipeline."
    icon="tabler:device-desktop"
    href="../stages/demo.html"
  />
</DocCards>
