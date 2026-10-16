---
tags:
- espnet
- espnet3
- ${system}
- ${corpus}
- voice-conversion
language: ${lang}
license: ${license}
---

# ESPnet3 ${system} model

${description}

This is a **kNN-VC** voice-conversion model: a HiFi-GAN vocoder trained on
(prematched) frozen WavLM features. The method and the training recipe are from
"Voice Conversion With Just Nearest Neighbors" by Matthew Baas, Benjamin van
Niekerk and Herman Kamper (Interspeech 2023,
[paper](https://arxiv.org/abs/2305.18975),
[code](https://github.com/bshall/knn-vc), MIT License); only the vocoder is
trained, the WavLM encoder is frozen and the kNN converter has no parameters.
Please cite their paper (below) when you use this model.

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

Voice conversion takes a source utterance plus one or more reference
utterances of the target speaker, and returns the converted waveform.

```python
from espnet3.publication import InferenceModel

${usage_load_call}
result = model(
    {
        "speech": source_waveform,          # 16 kHz mono numpy array
        "reference_speech": [ref1, ref2],   # target-speaker waveforms
        "target_speaker": "spk1",           # optional cache key
    }
)
converted = result["wav"]                   # 16 kHz mono numpy array
```

The sample is a dict because the model has several inputs; its keys are the
`input_key` entries of the bundled inference config.

The WavLM-Large checkpoint is fetched from the kNN-VC release on first use;
set `wavlm_checkpoint` in the bundled inference config to a local copy to
avoid the download.

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

### Citing kNN-VC

```
@inproceedings{baas2023knnvc,
  author={Matthew Baas and Benjamin van Niekerk and Herman Kamper},
  title={Voice Conversion With Just Nearest Neighbors},
  year={2023},
  booktitle={Proceedings of Interspeech}
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
