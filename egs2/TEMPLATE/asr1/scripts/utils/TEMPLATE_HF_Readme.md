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

\`\`\`bash
cd espnet
${_checkout}
pip install -e .
cd $(pwd | rev | cut -d/ -f1-3 | rev)
./run.sh --skip_data_prep false --skip_train true --download_model ${hf_repo}
\`\`\`

$(cat "${task_exp}"/RESULTS.md)

## ${espnet_task} config

<details><summary>expand</summary>

\`\`\`
$(cat "${task_exp}"/config.yaml)
\`\`\`

</details>

$(if [ -z ${var+use_lm} ]; then 
    ${use_lm} && echo "## LM config
    
<details><summary>expand</summary>

\`\`\`
  $(cat "${lm_exp}"/config.yaml)
\`\`\`

</details>
    ";
fi)

### Citing ESPnet

```BibTex
@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson {Enrique Yalta Soplin} and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={{ESPnet}: End-to-End Speech Processing Toolkit},
  year={2018},
  booktitle={Proceedings of Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1456}
}
<add_tts_reference>
```

or arXiv:

```bibtex
@misc{watanabe2018espnet,
      title={ESPnet: End-to-End Speech Processing Toolkit}, 
      author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson Enrique Yalta Soplin and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
      year={2018},
      eprint={1804.00015},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
