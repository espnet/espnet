# `espnet3.publication`

Minimal direct-inference API for published ESPnet models.

## Load a published model from model-zoo

```python
import soundfile as sf

from espnet3.publication import InferenceSession

session = InferenceSession.from_pretrained(
    "espnet/your-model-tag",
)

audio, sample_rate = sf.read("sample.wav", dtype="float32")
result = session(audio)
print(result)
```

When the published bundle contains an inference config, `InferenceSession`
loads that config first and instantiates `inference.yaml:model`.
If no published inference config is available, you can still fall back to an
explicit backend class.

## Enable bundled recipe user code

Use `trust_user_code=True` when the published inference config references
bundled modules under `src/`.

```python
from espnet3.publication import InferenceSession

session = InferenceSession.from_pretrained(
    "org/custom-model",
    trust_user_code=True,
)
```

If the bundle references `src.*` and `trust_user_code` is not enabled,
session creation fails instead of importing arbitrary code.

## Build from an inference config

```python
from omegaconf import OmegaConf

from espnet3.publication import InferenceSession

config = OmegaConf.create(
    {
        "device": "cpu",
        "input_key": "speech",
        "model": {
            "_target_": "espnet2.bin.asr_inference.Speech2Text",
            "asr_train_config": "exp/config.yaml",
            "asr_model_file": "exp/valid.acc.best.pth",
        },
    }
)

session = InferenceSession.from_config(config)
result = session(audio_array)
```

## Fallback to an explicit backend class

```python
session = InferenceSession.from_pretrained(
    "espnet/your-model-tag",
    backend_class="espnet2.bin.asr_inference.Speech2Text",
    enable_user_code=False,
)
```

## Structured samples and batched inference

```python
sample = {"speech": audio_array, "utt_id": "utt-0001"}
result = session.forward(sample, idx=0)

batch = session.forward_batch(
    [
        {"speech": audio_a, "utt_id": "utt-a"},
        {"speech": audio_b, "utt_id": "utt-b"},
    ]
)
```

`forward_batch()` can try a single batched backend call first and fall back
to per-sample execution if the backend does not support list inputs.
