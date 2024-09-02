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
actions:
  - text: Get Started
    icon: book
    link: ./get_started.md
    type: primary

  - text: Demos
    icon: lightbulb
    link: ./notebook/


highlights:
  - header: Providing different usage types
    bgImage: https://theme-hope-assets.vuejs.press/bg/3-light.svg
    bgImageDark: https://theme-hope-assets.vuejs.press/bg/3-dark.svg
    features:
      - title: "Inferencing existing ESPnet models"
        details: "<code>pip install espnet espnet-model-zoo</code> and use it straight away."
        icon: bolt-lightning
        link: ./notebook/inference.md
      - title: "Fine-tuning ESPnet models"
        details: "<code>pip install espnet</code> and use the <code>espnetez</code> module."
        icon: fire
        link: ./notebook/inference.md
      - title: "High-performance training and full experiment replication"
        details: "Go through full installation and leverage the existing recipes."
        icon: server
        link: ./espnet2_tutorial.md


  - header: Supports many tasks
    description: We provide a complete setup for various speech processing tasks.
    image: https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEj3mOiQTPh_S9XW6m94OQYjucUzUu7L9uEcHP9YsADUGWTcmscynkrLc1Zs8o5rA3G9lSNnEpyHBMCnZzBepYdW8jVofKnLflvOsu-ywIZpQf1Kw5l6tzvhEA1q2cbnFDIzIDlOUOKPOarf/s800/cooking_recipe.png
    bgImage: https://theme-hope-assets.vuejs.press/bg/2-light.svg
    bgImageDark: https://theme-hope-assets.vuejs.press/bg/2-dark.svg
    bgImageStyle:
      background-repeat: repeat
      background-size: initial
    features:
      - title: "ASR: Automatic Speech Recognition"
        link: https://github.com/espnet/espnet?tab=readme-ov-file#asr-automatic-speech-recognition

      - title: "TTS: Text-to-speech"
        link: https://github.com/espnet/espnet?tab=readme-ov-file#tts-text-to-speech

      - title: "SE: Speech enhancement (and separation)"
        link: https://github.com/espnet/espnet?tab=readme-ov-file#se-speech-enhancement-and-separation

      - title: "SUM: Speech Summarization"
        link: https://github.com/espnet/espnet?tab=readme-ov-file#sum-speech-summarization

      - title: "SVS: Singing Voice Synthesis"
        link: https://github.com/espnet/espnet?tab=readme-ov-file#svs-singing-voice-synthesis

      - title: "SSL: Self-supervised Learning"
        link: https://github.com/espnet/espnet?tab=readme-ov-file#ssl-self-supervised-learning

      - title: "UASR: Unsupervised ASR "
        details: "EURO: ESPnet Unsupervised Recognition - Open-source"
        link: https://github.com/espnet/espnet?tab=readme-ov-file#uasr-unsupervised-asr-euro-espnet-unsupervised-recognition---open-source

      - title: "S2T: Speech-to-text with Whisper-style multilingual   multitask models"
        link: https://github.com/espnet/espnet?tab=readme-ov-file#s2t-speech-to-text-with-whisper-style-multilingual-multitask-models

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
        link: ./tutorial.md

      - title: ESPnet1
        details: Documents on ESPnet1 recipes (Legacy)
        icon: graduation-cap
        link: ./espnet1_tutorial.md

      - title: Training configurations
        details: Understanding and updating the training configurations
        icon: sliders
        link: ./espnet2_training_option.md

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
