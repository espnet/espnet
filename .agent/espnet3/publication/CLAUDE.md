# `espnet3/publication/`

See [`.agent/CLAUDE.md`](../../CLAUDE.md) for cross-cutting guidance.

```
espnet3/publication/
├── inference_model.py        # InferenceModel -- loads a packed bundle (local dir or HF hub tag)
└── demo/
    ├── packing.py            # pack_demo() / upload_demo()
    ├── assets.py             # UIAsset(base) / UIAssetRegistry / DefaultAudioUI / DefaultTextUI
    └── session.py            # DemoSession / load_demo_session -- runtime wrapper used by the Gradio app
```

Packaging a trained model or a demo app for distribution (Hugging Face Hub or a local bundle). The
`pack_model`/`upload_model` free functions themselves live in
`espnet3/utils/publication_utils.py` (see [`.agent/espnet3/utils/CLAUDE.md`](../utils/CLAUDE.md)) --
this package covers packed-model *consumption* (`inference_model.py`) and demo packing/serving.

- **`inference_model.py`** -- `InferenceModel`: loads a bundle produced by `pack_model`
  -- either a local directory or a Hugging Face Hub tag -- and exposes
  a single callable inference API (`InferenceModel.from_pretrained(...)`, `model(**inputs)`). Also
  used by `ci/test_integration_espnet3_publication_check.py` to validate a packed bundle from outside
  the recipe tree (see [`.agent/ci/CLAUDE.md`](../../ci/CLAUDE.md)).
- **`demo/packing.py`** -- `pack_demo(system)` / `upload_demo(system)`: assembles a runnable demo
  bundle (app script, UI assets, a relative or bundled copy of the packed model) and pushes it to a
  Hugging Face Space.
- **`demo/assets.py`** -- `UIAsset` (base), `UIAssetRegistry`, `DefaultAudioUI`, `DefaultTextUI`:
  the built-in Gradio UI components a demo config can select via `_target_`.
- **`demo/session.py`** -- `DemoSession` / `load_demo_session`: the runtime object the packed
  `app.py` uses to load the model and dispatch a UI event to it.
