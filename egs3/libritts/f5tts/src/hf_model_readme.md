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
from espnet3.api.inference import load

model = load("${hf_repo}")
output = model(
    "The text to synthesize.",
    "reference.wav",  # the voice to clone: a path, or a (rate, samples) pair
    "The transcript of the reference recording.",
)
samples, sample_rate = output["wav"].array, output["wav"].rate
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

### Citing F5-TTS

Yushen Chen, Zhikang Niu, et al.
"F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching"
https://aclanthology.org/2025.acl-long.313/

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
