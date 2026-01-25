---
home: true
icon: /assets/image/espnet3.png
title: ESPnet3
author:
  name: "Masao Someki"
date: 2025-11-26
heroImage: /assets/image/espnet3-logo.png
heroImageStyle:
  - height: auto
  - width: 400px

heroText: "ESPnet3: a modern major release"
tagline: "Pythonic, end-to-end speech workflows—from dataset creation to training, inference, evaluation, packaging, and demo generation."

highlights:
  - header: Start with ESPnet3
    bgImage: https://theme-hope-assets.vuejs.press/bg/3-light.svg
    features:
      - title: "Getting Started"
        details: "Quick start for recipes and basic workflows."
        icon: fa-solid:rocket
        link: ./get_started.md
      - title: "Installation"
        details: "Set up ESPnet3 and dependencies."
        icon: fa-solid:download
        link: ./install.md
      - title: "Config overview"
        details: "How stage YAML configs are organized and used."
        icon: fa-solid:cog
        link: ./config/index.md

  - header: Stages
    bgImage: https://theme-hope-assets.vuejs.press/bg/2-light.svg
    bgImageStyle:
      background-repeat: repeat
      background-size: initial
    features:
      - title: "create_dataset"
        link: ./stages/create-dataset.md
        details: "Download/build datasets for your recipe."
        icon: fa-solid:database
      - title: "collect_stats"
        link: ./stages/collect-stats.md
        details: "Compute feature shapes and global stats."
        icon: fa-solid:chart-line
      - title: "train"
        link: ./stages/train.md
        details: "Run Lightning training with `train.yaml`."
        icon: fa-solid:graduation-cap
      - title: "infer"
        link: ./stages/inference.md
        details: "Write `.scp` outputs under `infer_dir`."
        icon: fa-solid:bolt
      - title: "measure"
        link: ./stages/measure.md
        details: "Compute metrics from inference outputs."
        icon: fa-solid:ruler-combined
      - title: "Publish-related"
        link: ./stages/publish.md
        details: "Pack and upload model artifacts (`pack_model` / `upload_model`)."
        icon: fa-solid:box-archive
      - title: "Demo stages"
        link: ./stages/demo.md
        details: "Generate and upload a demo UI."
        icon: fa-solid:display
      - title: "System-specific stages"
        link: ./stages/system-specific.md
        details: "Add your own stages in the System class."
        icon: fa-solid:diagram-project

  - header: Core packages
    bgImage: https://theme-hope-assets.vuejs.press/bg/4-light.svg
    bgImageStyle:
      background-repeat: repeat
      background-size: initial
    features:
      - title: "Systems"
        link: ./core/systems.md
        details: "`espnet3/systems`: stage orchestration and task entry points."
        icon: fa-solid:diagram-project
      - title: "Components"
        link: ./core/components.md
        details: "`espnet3/components`: reusable data/training/model/metric blocks."
        icon: fa-solid:cubes
      - title: "Parallel"
        link: ./core/parallel.md
        details: "`espnet3/parallel`: Provider/Runner execution stack."
        icon: fa-solid:server
      - title: "Demo"
        link: ./core/demo.md
        details: "`espnet3/demo`: packing, runtime, and UI wiring."
        icon: fa-solid:display

footer: Apache License 2.0, Copyright © 2024-present ESPnet community
---

## How to cite ESPnet
```
@inproceedings{watanabe18_interspeech,
  title     = {ESPnet: End-to-End Speech Processing Toolkit},
  author    = {Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson {Enrique Yalta Soplin} and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  year      = {2018},
  booktitle = {Proc. Interspeech},
  pages     = {2207--2211},
  doi       = {10.21437/Interspeech.2018-1456},
  issn      = {2958-1796},
}
```
To cite individual modules, models, or recipes, please refer to [Additional Citations](./citations.md).
