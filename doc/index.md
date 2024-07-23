git s---
home: true
icon: /assets/image/espnet.png
title: ESPnet
heroImage: /assets/image/espnet_logo1.png
heroImageStyle:
  - width: 80%
heroText: "ESPnet: end-to-end speech processing toolkit"
tagline: "ESPnet is an end-to-end speech processing toolkit covering many speech-related tasks."
actions:
  - text: Get Started
    icon: book
    link: ./espnet2_tutorial.md
    type: primary

  - text: Demos
    icon: lightbulb
    link: ./notebook/


highlights:
  - header: Easy to install
    image: /assets/image/box.svg
    bgImage: https://theme-hope-assets.vuejs.press/bg/3-light.svg
    bgImageDark: https://theme-hope-assets.vuejs.press/bg/3-dark.svg
    highlights:
      - title: <code>pip install espnet</code> for easy install.
      - title: See <a href="./installation.md">this instruction</a> for more details.

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

  - header: More Documents
    description: You can find tutorials on ESPnet packages.
    bgImage: https://theme-hope-assets.vuejs.press/bg/10-light.svg
    bgImageDark: https://theme-hope-assets.vuejs.press/bg/10-dark.svg
    bgImageStyle:
      background-repeat: repeat
      background-size: initial
    highlights:
      - title: Tutorial
        icon: /assets/icon/school_24dp_5F6368_FILL0_wght400_GRAD0_opsz24.svg
        link: ./tutorial.md

      - title: Tutorial (ESPnet1)
        icon: /assets/icon/school_24dp_5F6368_FILL0_wght400_GRAD0_opsz24.svg
        link: ./espnet1_tutorial.md

      - title: Training Config
        icon: sliders
        link: ./espnet2_training_option.md

      - title: Format audio to wav.scp
        icon: arrow-rotate-right
        link: ./espnet2_format_wav_scp.md

      - title: Task class and data
        icon: database
        link: ./espnet2_task.md

      - title: Docker
        icon: /assets/icon/docker-mark-blue.svg
        link: ./docker.md

      - title: Job scheduling system
        icon: server
        link: ./parallelization.md

      - title: Distributed training
        icon: server
        link: ./espnet2_distributed.md

      - title: Document Generation
        icon: book
        link: ./document.md

footer: Apache License 2.0, Copyright © 2024-present ESPnet community
---

## Citations

```
@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson {Enrique Yalta Soplin} and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={{ESPnet}: End-to-End Speech Processing Toolkit},
  year={2018},
  booktitle={Proceedings of Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1456}
}
@inproceedings{hayashi2020espnet,
  title={{Espnet-TTS}: Unified, reproducible, and integratable open source end-to-end text-to-speech toolkit},
  author={Hayashi, Tomoki and Yamamoto, Ryuichi and Inoue, Katsuki and Yoshimura, Takenori and Watanabe, Shinji and Toda, Tomoki and Takeda, Kazuya and Zhang, Yu and Tan, Xu},
  booktitle={Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7654--7658},
  year={2020},
  organization={IEEE}
}
@inproceedings{inaguma-etal-2020-espnet,
    title = "{ESP}net-{ST}: All-in-One Speech Translation Toolkit",
    author = "Inaguma, Hirofumi  and
      Kiyono, Shun  and
      Duh, Kevin  and
      Karita, Shigeki  and
      Yalta, Nelson  and
      Hayashi, Tomoki  and
      Watanabe, Shinji",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-demos.34",
    pages = "302--311",
}
@article{hayashi2021espnet2,
  title={Espnet2-tts: Extending the edge of tts research},
  author={Hayashi, Tomoki and Yamamoto, Ryuichi and Yoshimura, Takenori and Wu, Peter and Shi, Jiatong and Saeki, Takaaki and Ju, Yooncheol and Yasuda, Yusuke and Takamichi, Shinnosuke and Watanabe, Shinji},
  journal={arXiv preprint arXiv:2110.07840},
  year={2021}
}
@inproceedings{li2020espnet,
  title={{ESPnet-SE}: End-to-End Speech Enhancement and Separation Toolkit Designed for {ASR} Integration},
  author={Chenda Li and Jing Shi and Wangyou Zhang and Aswin Shanmugam Subramanian and Xuankai Chang and Naoyuki Kamo and Moto Hira and Tomoki Hayashi and Christoph Boeddeker and Zhuo Chen and Shinji Watanabe},
  booktitle={Proceedings of IEEE Spoken Language Technology Workshop (SLT)},
  pages={785--792},
  year={2021},
  organization={IEEE},
}
@inproceedings{arora2021espnet,
  title={{ESPnet-SLU}: Advancing Spoken Language Understanding through ESPnet},
  author={Arora, Siddhant and Dalmia, Siddharth and Denisov, Pavel and Chang, Xuankai and Ueda, Yushi and Peng, Yifan and Zhang, Yuekai and Kumar, Sujay and Ganesan, Karthik and Yan, Brian and others},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7167--7171},
  year={2022},
  organization={IEEE}
}
@inproceedings{shi2022muskits,
  author={Shi, Jiatong and Guo, Shuai and Qian, Tao and Huo, Nan and Hayashi, Tomoki and Wu, Yuning and Xu, Frank and Chang, Xuankai and Li, Huazhe and Wu, Peter and Watanabe, Shinji and Jin, Qin},
  title={{Muskits}: an End-to-End Music Processing Toolkit for Singing Voice Synthesis},
  year={2022},
  booktitle={Proceedings of Interspeech},
  pages={4277-4281},
  url={https://www.isca-speech.org/archive/pdfs/interspeech_2022/shi22d_interspeech.pdf}
}
@inproceedings{lu22c_interspeech,
  author={Yen-Ju Lu and Xuankai Chang and Chenda Li and Wangyou Zhang and Samuele Cornell and Zhaoheng Ni and Yoshiki Masuyama and Brian Yan and Robin Scheibler and Zhong-Qiu Wang and Yu Tsao and Yanmin Qian and Shinji Watanabe},
  title={{ESPnet-SE++: Speech Enhancement for Robust Speech Recognition, Translation, and Understanding}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={5458--5462},
}
@article{gao2022euro,
  title={{EURO}: {ESPnet} Unsupervised ASR Open-source Toolkit},
  author={Gao, Dongji and Shi, Jiatong and Chuang, Shun-Po and Garcia, Leibny Paola and Lee, Hung-yi and Watanabe, Shinji and Khudanpur, Sanjeev},
  journal={arXiv preprint arXiv:2211.17196},
  year={2022}
}
@article{peng2023reproducing,
  title={Reproducing Whisper-Style Training Using an Open-Source Toolkit and Publicly Available Data},
  author={Peng, Yifan and Tian, Jinchuan and Yan, Brian and Berrebbi, Dan and Chang, Xuankai and Li, Xinjian and Shi, Jiatong and Arora, Siddhant and Chen, William and Sharma, Roshan and others},
  journal={arXiv preprint arXiv:2309.13876},
  year={2023}
}
```