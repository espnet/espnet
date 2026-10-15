# Direct backend inference

ESPnet3 recipes build a backend directly and pass configured dataset columns
to its callable. There is no shared inference base class, field registry,
implicit streaming conversion, or declaration-driven model loader.

For an ASR recipe, configure the native backend and result formatter:

```yaml
model:
  _target_: espnet2.bin.asr_inference.Speech2Text
  asr_train_config: ${exp_dir}/config.yaml
  asr_model_file: ${exp_dir}/last.ckpt
  beam_size: 1
input_key: speech
output_fn: egs3.TEMPLATE.esp2_asr.src.inference.build_output
```

The provider constructs `model` on the worker's device. The Runner retrieves
`input_key` from each dataset item, calls the backend, and passes its native
result to `output_fn(data=..., model_output=..., idx=...)`. The formatter
returns a record containing `utt_id` and the output values to write as SCPs.
`output_artifacts` configures serialization of arrays and other artifacts;
a waveform's sample rate is an explicit writer setting.

Use `espnet2.bin.asr_transducer_inference.Speech2Text` with
`return_decoded_hyp: true` for the separate transducer task. Native backend
return types and streaming support remain those of the backend; callers do
not receive automatically derived single, batch, or streaming methods.

Recipe datasets retain `BaseDataset` and declare Python types, for example
`fields = {"speech": np.ndarray, "text": str}`. The first item is checked for required keys and Python types.
Include `type(None)` in a tuple of accepted types for optional keys.
Transforms and preprocessors remain recipe configuration.

Metrics implement `update()` and `compute()` and can retain their existing
registered aggregation state. Declare `inputs = {"ref": str, "hyp": str}`
and `outputs = ("WER",)` on the metric. Contracts check required input names,
aligned batch lengths, element types, and numeric results, including direct
`update()`/`compute()` calls. Configure their keyword arguments explicitly:

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.esp2_asr.metrics.wer.WER
    inputs:
      ref: dataset:text
      hyp: text
```

This reads the reference from the dataset and the hypothesis from `text.scp`.
Measurement validates metric input bindings without inspecting the model class.
SCP alignment and the metric's own contracts are checked independently of
inference.

Model packing and uploading retain checkpoints, configuration and metadata.
Archives are used through their recipe and native backend, rather than a
unified callable loader. Old configurations naming removed wrapper classes
must be updated to the native target and explicit input/output configuration.

`pack_demo` and `upload_demo` remain available. The ASR template supplies
`src/app.py`, which reads the packed `conf/inference.yaml` and directly builds
the native ASR or transducer backend. Set `model.sample_rate` to the training
audio rate. A recipe can override `demo_config.ui.app_script`; packing copies the app, model reference, extra
assets, requirements and optional Space card, and uploading publishes that
directory. UI generation from model declarations and notebook generation are
removed. Standalone applications and notebooks call the native backend
themselves. The existing ESPnet2 task commands, demo and MCP tools remain
independent clients of those native backends.

The independent helpers in `espnet3.publication.demo` remain available:
packing/uploading, the UI asset registry, and `DemoSession` for configuration,
description text, component construction and button wiring. Call
`load_demo_session(demo_dir, "demo.yaml", model=decode)` with a callable built
by the recipe and explicit `ui.inputs`/`ui.outputs`. The callable returns
named values in Gradio-compatible formats. Sessions do not construct models
or derive UI specs from model declarations.
