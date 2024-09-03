---
home: true
icon: /assets/image/espnet.png
title: ESPnet
heroImage: /assets/image/espnet_logo1.png
heroImageStyle:
  - height: auto
  - width: 400px

heroText: "End-to-end Speech Processing toolkit"
tagline: "ESPnet is the state-of-the-art toolkit that covers end-to-end speech recognition, text-to-speech, speech translation, speech enhancement, speaker diarization, spoken language understanding, and much more!"


highlights:
  - header: Get started with ESPnet!
    bgImage: https://theme-hope-assets.vuejs.press/bg/3-light.svg
    bgImageDark: https://theme-hope-assets.vuejs.press/bg/3-dark.svg
    features:
      - title: "Inferencing existing ESPnet models"
        details: "<code>pip install espnet espnet-model-zoo</code> and use it straight away."
        icon: bolt-lightning
        link: notebook/#demo
      - title: "Fine-tuning ESPnet models"
        details: "<code>pip install espnet</code> and use the <code>espnetez</code> module."
        icon: fire
        link: notebook/#espnet-ez
      - title: "High-performance training and full experiment replication"
        details: "Go through full installation and leverage the existing recipes."
        icon: server
        link: ./espnet2_tutorial.md


  - header: Extensive task coverage
    description: We provide a complete recipe for various speech processing tasks.
    image: https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEj3mOiQTPh_S9XW6m94OQYjucUzUu7L9uEcHP9YsADUGWTcmscynkrLc1Zs8o5rA3G9lSNnEpyHBMCnZzBepYdW8jVofKnLflvOsu-ywIZpQf1Kw5l6tzvhEA1q2cbnFDIzIDlOUOKPOarf/s800/cooking_recipe.png
    bgImage: https://theme-hope-assets.vuejs.press/bg/2-light.svg
    bgImageDark: https://theme-hope-assets.vuejs.press/bg/2-dark.svg
    bgImageStyle:
      background-repeat: repeat
      background-size: initial
    features:
      - title: "ASR: Automatic Speech Recognition"
        link: ./recipe/asr1.md

      - title: "TTS: Text-to-speech"
        link: ./recipe/tts1.md

      - title: "Speech Enhancement"
        link: ./recipe/enh1.md

      - title: "Weakly-supervised Learning"
        link: ./recipe/s2t1.md

      - title: "Speaker Embedding"
        link: ./recipe/spk1.md

      # - title: "SSL: Self-supervised Learning"
      #   link: ./recipe/ssl1.md

      # - title: "Speech Language Model"
      #   link: ./recipe/speechlm1.md

      # - title: "SLU: Spoken Language Understanding"
      #   link: ./recipe/slu1.md

      - title: "Speech-to-Text Translation"
        link: ./recipe/st1.md

      # - title: "Speech-to-Speech Translation"
      #   link: ./recipe/s2st1.md

      - title: "Speech Codec"
        link: ./recipe/codec1.md

      - title: "... And much more!"
        link: recipe/

  - header: Tutorials
    bgImage: https://theme-hope-assets.vuejs.press/bg/10-light.svg
    bgImageDark: https://theme-hope-assets.vuejs.press/bg/10-dark.svg
    bgImageStyle:
      background-repeat: repeat
      background-size: initial
    features:
      - title: ESPnet2
        details: Leveraging ESPnet2 recipes for full replication
        icon: graduation-cap
        link: ./espnet2_tutorial.md

      - title: ESPnet1
        details: Documents on ESPnet1 recipes (Legacy)
        icon: graduation-cap
        link: ./espnet1_tutorial.md

      - title: Training configurations
        details: Understanding and updating the training configurations
        icon: sliders
        link: ./espnet2_training_option.md

      - title: Recipe tips
        details: Various tips on using run.sh in ESPnet recipes
        icon: clipboard-check
        link: ./tutorial.md

      - title: Audio formatting
        details: Formatting audios into wav.scp for ESPnet recipes
        icon: microphone
        link: ./espnet2_format_wav_scp.md

      - title: Task class and data input system
        details: Common task/data interface for ESPnet2
        icon: database
        link: ./espnet2_task.md

      - title: Docker
        details: Running ESPnet on Docker
        icon: fa-brands fa-docker
        link: ./docker.md

      - title: Job scheduling system
        details: Distributing jobs in multi-machine environment within recipes
        icon: list-check
        link: ./parallelization.md

      - title: Distributed training
        details: Handling multiple GPUS for training
        icon: network-wired
        link: ./espnet2_distributed.md

      - title: Document Generation
        details: Details on fixing the development document
        icon: book
        link: ./document.md

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
To additionally cite individual modules, models, or recipes, please refer to [Additional Citations](./citations.md).
