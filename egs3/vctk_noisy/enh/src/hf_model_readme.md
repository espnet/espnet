---
tags:
- espnet
- ${system}
- ${corpus}
language: ${lang}
license: ${license}
---

# ESPnet3 ${system} model

${description}

## Model

- Repository: `${hf_repo}`
- Recipe: `${recipe}`
- Corpus: `${corpus}`
- System: `${system}`
- Creator: `${creator}`
- Created: `${created_at}`
- Branch: `${git_branch}`
- Git: `${git_head}` (${git_dirty})
- Origin: ${git_origin}

${model_summary_section}
${model_detail_section}

## Usage

```python
from espnet3.publication import InferenceModel

${usage_load_call}
result = model(sample)
```

## Packaging

- Bundle: `${pack_name}`
- Exp dir: `${exp_dir}`
- Strategy: `${pack_strategy}`

${results_section}
${results_note}

## Training config

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
