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

Datasets can be ordinary Python or PyTorch datasets returning mappings.
They no longer need a toolkit-specific superclass or item declarations.
Transforms and preprocessors remain recipe configuration.

Metrics implement `update()` and `compute()` and can retain their existing
registered aggregation state. Configure their keyword arguments explicitly:

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.esp2_asr.metrics.wer.WER
    inputs:
      ref: dataset:text
      hyp: text
```

This reads the reference from the dataset and the hypothesis from `text.scp`.
Measurement does not inspect a model class or validate an input/output schema.
Ordinary missing-input, SCP-alignment, and backend errors still surface.

Model packing and uploading retain checkpoints, configuration and metadata.
Archives are used through their recipe and native backend, rather than a
unified callable loader. Old configurations naming removed wrapper classes
must be updated to the native target and explicit input/output configuration.

Generated demo sessions, demo packing/uploading stages and notebook generation
are removed. Standalone applications and notebooks can call the native
backend themselves. The existing ESPnet2 task commands, demo and MCP tools
remain independent clients of those native backends.
