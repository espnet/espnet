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
        "spembs": spembs,  # the speaker x-vector, a 1-D array
    }
)
audio = output["wav"]  # an Audio: audio.array (float32) at audio.rate
```

`spembs` must come from the extractor the model was trained with
(`xvector.spk_embed_tag` in `conf/training.yaml`); the demo launcher in
`egs3/TEMPLATE/esp2_gan_tts/src/app.py` computes it from a reference
recording.

## Packaging

- Bundle: `${pack_name}`
- Exp dir: `${exp_dir}`
- Strategy: `${pack_strategy}`

${results_section}
${results_note}

## Training config

<details><summary>expand</summary>

```yaml
${train_config}
```

</details>

### Citing ESPnet

```bibtex
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
