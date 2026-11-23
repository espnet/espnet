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

model = InferenceModel.from_pretrained("${hf_repo}")
output = model(
    {
        "text": "The text to synthesize.",
        "reference_speech": "reference.wav",  # a path, or a (rate, samples) pair
        "reference_text": "The reference transcript.",
    }
)
audio = output["wav"]  # an Audio: audio.array (float32) at audio.rate
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

```
@inproceedings{chen-etal-2025-f5,
  author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang
    and Jian Zhao and Kai Yu and Xie Chen},
  title={{F5-TTS}: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching},
  year={2025},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for
    Computational Linguistics (Volume 1: Long Papers)},
  pages={6255--6271},
  doi={10.18653/v1/2025.acl-long.313}
}
```

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
