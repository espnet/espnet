---
tags:
- espnet
- speech-enhancement
- ${system}
- ${corpus}
license: ${license}
---

# ESPnet3 SNOAR speech enhancement model

${description}

## Model

- Repository: `${hf_repo}`
- Recipe: `${recipe}`
- System: `${system}`
- Created: `${created_at}`
- Git: `${git_head}` (${git_dirty})

${model_summary_section}
${model_detail_section}

## Usage

```bash
espnet snoar enhance noisy.wav -o enhanced.wav --model ${hf_repo}
```

```python
from espnet3.publication import InferenceModel

model = InferenceModel.from_pretrained("${hf_repo}")
enhanced = model({"speech": (16000, waveform)})["enhanced"].array
```

The network is the bundled `export` directory (`config.yaml`, `model.pth`,
`snoar.yaml`): an ordinary ESPnet2 enhancement model plus the SNOAR readout.

${results_section}
${results_note}

## Training config

See `conf/training.yaml`.
