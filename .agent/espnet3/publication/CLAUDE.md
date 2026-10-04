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

## Publication configuration and stage flow

Publication is driven by the five recipe configs loaded by `run.py`, not by a separate publication
CLI. Pass the relevant configs explicitly:

```bash
python run.py --stages pack_model \
  --training_config conf/tuning/training_e_branchformer.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml \
  --publication_config conf/publication.yaml
```

`run.py` propagates the training identity (`exp_tag`, `exp_dir`, and related paths) into the
publication config before the stage runs. `publication.yaml` then controls `pack_model.out_dir`,
the `include`/`exclude` file set, named artifacts, and the README metadata. Add recipe-local Python
code to `pack_model.include` when the packed inference config refers to it. The usual flow is
`infer` -> `measure` -> `pack_model` -> optional `upload_model`; this lets the packer include the
inference results and `metrics.json` in the model README.

`InferenceModel.from_packed()` is the external consumer of the resulting bundle. It loads the
packed `conf/inference.yaml`, reconstructs the configured backend, and calls the same optional
recipe `output_fn` used by inference. Keep the packed bundle self-contained and use
`trust_user_code` only when bundled recipe code is intentionally trusted.

## Demo session design

The demo has three layers with separate responsibilities:

1. `demo.yaml` declares the model reference (`model.dir_or_tag`), model call-time kwargs
   (`model.call_args`), UI input/output specs, and the app script to copy.
2. `load_demo_session()` loads that config and constructs a `DemoSession`. The session resolves the
   packed model through `InferenceModel`, resolves UI assets, and creates the inference callable.
3. The packed recipe `app.py` owns the Gradio layout. The default app builds components from
   `session.input_specs`/`output_specs`, passes their values positionally to
   `session.create_inference_fn()`, and displays the returned values.

The runtime inference path is therefore:

```text
Gradio values -> input spec keys -> InferenceModel(model_input, **call_args)
              -> result keys -> output spec values -> Gradio components
```

ESPnet3 owns model loading, packed-config resolution, input-key mapping, call-time arguments, and
the generic session callable. The recipe owns the UI layout and any presentation-specific logic.

For ordinary UI changes, edit `demo.yaml`'s `ui.inputs`/`ui.outputs` (the `key` values must match
the model input/output contract) and reuse the built-in `audio`/`text` assets. For a new reusable
component type, subclass `UIAsset` and register it in `DEFAULT_UI_ASSETS` in `demo/assets.py`.
For a one-off layout, edit or replace the recipe's `ui.app_script` and call the session yourself;
keep the positional order of Gradio inputs and outputs aligned with the specs. After changing the
app or UI assets, run `pack_demo` again so the modified script and config are copied into the packed
demo, then use the demo integration test/UI smoke test to verify the complete click-to-inference
path.
