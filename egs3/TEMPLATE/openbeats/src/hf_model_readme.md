---
tags:
- espnet
- espnet3
- ${system}
- ${corpus}
- audio
- self-supervised-learning
license: ${license}
---

# ESPnet3 ${system} model

${description}

This is a **BEATs** audio encoder pre-trained with iterative masked token
prediction ("BEATs: Audio Pre-Training with Acoustic Tokenizers", Chen et al.,
ICML 2023, [paper](https://arxiv.org/abs/2212.09058)) using the OpenBEATs
recipe of ESPnet. It is a pre-trained encoder without a task head: fine-tune it
or probe it on a downstream task.

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

The bundle contains `exp/<exp_tag>/beats_encoder_iter<N>.pt`, a portable
checkpoint in the `{"model": ..., "cfg": ...}` layout that `BeatsEncoder`
loads directly (`<exp_tag>` is `beats_iter<N>_<ssl_tag>`, see the training
config below):

```python
from huggingface_hub import snapshot_download

from espnet2.beats.encoder import BeatsEncoder

model_dir = snapshot_download("${hf_repo}")
encoder = BeatsEncoder(
    input_size=1,
    beats_ckpt_path=f"{model_dir}/exp/<exp_tag>/beats_encoder_iter<N>.pt",
)
```

In an ESPnet downstream config, point the encoder at the same file:

```yaml
encoder: beats
encoder_conf:
  beats_ckpt_path: /path/to/beats_encoder_iter<N>.pt
```

## Packaging

- Bundle: `${pack_name}`
- Exp dir: `${exp_dir}`
- Strategy: `${pack_strategy}`

## Tokenizer codebook usage

`measure` reports how the tokenizer of this iteration uses its codebook on the
pre-training targets (usage, entropy in bits, perplexity).

${results_section}
${results_note}

## Training config

<details><summary>expand</summary>

```
${train_config}
```

</details>

### Citing BEATs

```
@inproceedings{chen2023beats,
  title={{BEATs}: Audio Pre-Training with Acoustic Tokenizers},
  author={Chen, Sanyuan and Wu, Yu and Wang, Chengyi and Liu, Shujie and
    Tompkins, Daniel and Chen, Zhuo and Che, Wanxiang and Yu, Xiangzhan and
    Wei, Furu},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  year={2023}
}
```

### Citing ESPnet

```
@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and
    Jiro Nishitoba and Yuya Unno and Nelson {Enrique Yalta Soplin} and Jahn Heymann
    and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={{ESPnet}: End-to-End Speech Processing Toolkit},
  year={2018},
  booktitle={Proceedings of Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1456}
}
```
