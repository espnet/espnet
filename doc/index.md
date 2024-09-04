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
    features:
      - title: "Running inference on existing ESPnet models"
        details: "<code>pip install espnet espnet-model-zoo</code> and start using it immediately."
        icon: fa-solid:bolt
        link: notebook/#demo
      - title: "Fine-tuning ESPnet models"
        details: "<code>pip install espnet</code> and use the <code>espnetez</code> module for fine-tuning."
        icon: fa-solid:fire
        link: notebook/#espnet-ez
      - title: "High-performance training and full experiment replication"
        details: "Complete the full installation and use the existing recipes."
        icon: fa-solid:server
        link: ./espnet2_tutorial.md

  - header: Comprehensive Task Coverage
    description: We offer complete recipes for a wide range of speech processing tasks.
    bgImage: https://theme-hope-assets.vuejs.press/bg/2-light.svg
    bgImageStyle:
      background-repeat: repeat
      background-size: initial
    features:
      - title: "ASR: Automatic Speech Recognition"
        link: ./recipe/asr1.md
        icon: material-symbols-light:speech-to-text-rounded

      - title: "TTS: Text-to-speech"
        link: ./recipe/tts1.md
        icon: material-symbols-light:text-to-speech-rounded

      - title: "Speech Enhancement"
        link: ./recipe/enh1.md
        icon: material-symbols:adaptive-audio-mic-rounded

      - title: "Weakly-supervised Learning"
        link: ./recipe/s2t1.md
        icon: guidance:children-must-be-supervised

      - title: "Speaker Embedding"
        link: ./recipe/spk1.md
        icon: ic:sharp-safety-divider

      - title: "Speech-to-Text Translation"
        link: ./recipe/st1.md
        icon: ph:translate

      - title: "Singing Voice Synthesis"
        link: ./recipe/svs1.md
        icon: mdi:music

      - title: "Discrete Unit ASR"
        link: ./recipe/asr2.md
        icon: carbon:string-integer

      - title: "Speech Codec"
        link: ./recipe/codec1.md
        icon: material-symbols:hd-outline

      - title: "ASR with Speech Enhancement"
        link: ./recipe/enh_asr1.md
        icon: icon-park-outline:voice-one

      - title: "... And much more!"
        link: recipe/

  - header: Tutorials
    bgImage: https://theme-hope-assets.vuejs.press/bg/4-light.svg
    bgImageStyle:
      background-repeat: repeat
      background-size: initial
    features:
      - title: ESPnet2
        details: Leveraging ESPnet2 recipes for full replication
        icon: mdi:graduation-cap
        link: ./espnet2_tutorial.md

      - title: ESPnet1
        details: Documents on ESPnet1 recipes (Legacy)
        icon: mdi:graduation-cap
        link: ./espnet1_tutorial.md

      - title: Training configurations
        details: Understanding and updating training configurations
        icon: fa6-solid:sliders
        link: ./espnet2_training_option.md

      - title: Recipe tips
        details: Various tips for using <code>run.sh</code> in ESPnet recipes
        icon: mdi:clipboard-check-outline
        link: ./tutorial.md

      - title: Audio formatting
        details: Formatting audio files into <code>wav.scp</code> for ESPnet recipes
        icon: mdi:microphone
        link: ./espnet2_format_wav_scp.md

      - title: Task class and data input system
        details: Common task/data interface for ESPnet2
        icon: material-symbols-light:database-outline
        link: ./espnet2_task.md

      - title: Docker
        details: Running ESPnet with Docker
        icon: mdi:docker
        link: ./docker.md

      - title: Job scheduling system
        details: Distributing jobs in a multi-machine environment
        icon: tabler:list-check
        link: ./parallelization.md

      - title: Distributed training
        details: Handling multiple GPUs for training
        icon: fa-solid:network-wired
        link: ./espnet2_distributed.md

      - title: Document Generation
        details: Details on fixing the ESPnet documentation
        icon: fa-solid:book
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
To cite individual modules, models, or recipes, please refer to [Additional Citations](./citations.md).
