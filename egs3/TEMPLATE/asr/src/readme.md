---
tags:
- espnet
- audio
language: ${lang}
license: ${license}
---

# ESPnet3 ${task} model

Repository: `${hf_repo}`

${description}

## Training

- System: `${system}`
- Recipe: `${recipe}`
- Creator: `${creator}`
- Created: `${created_at}`
- Git: `${git_head}` (${git_dirty})

## Pack

- Archive: `${pack_name}`
- Strategy: `${pack_strategy}`
- Exp dir: `${exp_dir}`

${results_section}

## Train config

<details><summary>expand</summary>

```
${train_config}
```

</details>

### Citing ESPnet

```
@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and
    Jiro Nishitoba and Yuya Unno and Nelson Yalta and Jahn Heymann and Matthew Wiesner
    and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={{ESPnet}: End-to-End Speech Processing Toolkit},
  year={2018},
  booktitle={Proceedings of Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456}
}
```
