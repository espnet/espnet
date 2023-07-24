---
tags:
- espnet
- audio
- ${hf_task}
language: ${lang}
datasets:
- ${_corpus}
license: cc-by-4.0
---

## ESPnet2 ${espnet_task} model

### \`${hf_repo}\`

This model was trained by ${_creator_name} using ${_task} recipe in [espnet](https://github.com/espnet/espnet/).

### Demo: How to use in ESPnet2

Follow the [ESPnet installation instructions](https://espnet.github.io/espnet/installation.html)
if you haven't done that already.

\`\`\`bash
cd espnet
${_checkout}
pip install -e .
cd $(pwd | rev | cut -d/ -f1-3 | rev)
./run.sh --skip_data_prep false --skip_train true --download_model ${hf_repo}
\`\`\`

$(if [ -f "${task_exp}"/RESULTS.md ]; then
  cat "${task_exp}"/RESULTS.md;
fi)

## ${espnet_task} config

<details><summary>expand</summary>

\`\`\`
$(cat "${task_exp}"/config.yaml)
\`\`\`

</details>

$(if [ -n "${var+use_lm}" ]; then
  ${use_lm} && echo "## LM config

<details><summary>expand</summary>

\`\`\`
  $(cat "${lm_exp}"/config.yaml)
\`\`\`

</details>
    ";
fi)

### Citing ESPnet

\`\`\`BibTex
@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson Yalta and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={{ESPnet}: End-to-End Speech Processing Toolkit},
  year={2018},
  booktitle={Proceedings of Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1456}
}

$(if [ "${espnet_task}" == "ENH" ]; then
  echo '
@inproceedings{ESPnet-SE,
  author = {Chenda Li and Jing Shi and Wangyou Zhang and Aswin Shanmugam Subramanian and Xuankai Chang and
  Naoyuki Kamo and Moto Hira and Tomoki Hayashi and Christoph B{"{o}}ddeker and Zhuo Chen and Shinji Watanabe},
  title = {ESPnet-SE: End-To-End Speech Enhancement and Separation Toolkit Designed for {ASR} Integration},
  booktitle = {{IEEE} Spoken Language Technology Workshop, {SLT} 2021, Shenzhen, China, January 19-22, 2021},
  pages = {785--792},
  publisher = {{IEEE}},
  year = {2021},
  url = {https://doi.org/10.1109/SLT48900.2021.9383615},
  doi = {10.1109/SLT48900.2021.9383615},
  timestamp = {Mon, 12 Apr 2021 17:08:59 +0200},
  biburl = {https://dblp.org/rec/conf/slt/Li0ZSCKHHBC021.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}';
fi)

$(if [ "${espnet_task}" == "TTS" ]; then
  echo '
@inproceedings{hayashi2020espnet,
  title={{Espnet-TTS}: Unified, reproducible, and integratable open source end-to-end text-to-speech toolkit},
  author={Hayashi, Tomoki and Yamamoto, Ryuichi and Inoue, Katsuki and Yoshimura, Takenori and Watanabe, Shinji and Toda, Tomoki and Takeda, Kazuya and Zhang, Yu and Tan, Xu},
  booktitle={Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7654--7658},
  year={2020},
  organization={IEEE}
}';
fi)

$(if [ "${espnet_task}" == "SVS" ]; then
  echo '
@inproceedings{shi22d_interspeech,
  author={Jiatong Shi and Shuai Guo and Tao Qian and Tomoki Hayashi and Yuning Wu and Fangzheng Xu and Xuankai Chang and Huazhe Li and Peter Wu and Shinji Watanabe and Qin Jin},
  title={{Muskits: an End-to-end Music Processing Toolkit for Singing Voice Synthesis}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={4277--4281},
  doi={10.21437/Interspeech.2022-10039}
}';
fi)
\`\`\`

or arXiv:

\`\`\`bibtex
@misc{watanabe2018espnet,
  title={ESPnet: End-to-End Speech Processing Toolkit},
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson Yalta and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  year={2018},
  eprint={1804.00015},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
\`\`\`
