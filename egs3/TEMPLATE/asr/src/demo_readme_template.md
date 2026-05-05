# ${title}

${description}

## Try it out

Use the microphone or upload an audio file above to transcribe speech.

## Programmatic usage

```python
from espnet3.publication import InferenceModel

${usage_load_call}
result = model({"speech": "/path/to/audio.wav"})
print(result["hyp"])
```

## Model info

- System: `${system}`
- Model: `${model_ref}`
- Recipe: `${recipe}`
- Creator: `${creator}`

## Citation

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
