# ASR stage configuration regression fixtures

`conf/` contains role configs, `snapshots.yaml` contains full saved stage
snapshots, and `cases.yaml` lists every ASR stage's directory key and CLI flag.
`__ROOT__` is replaced with a fresh temporary directory for each test.

The training and inference configs deliberately override `model` and `dataset`.
Those are normal stage-specific settings, not evidence of conflicting runs.
Snapshots are materialized in both the full-snapshot and baked-config layouts.
Tests verify resolution and persistence; stage bodies are mocked, so no training,
network upload, or model inference is performed. Missing-artifact validation in
actual model/data loaders remains covered by their own tests.

The tests express the intended behavior. They are not marked xfail when an
outstanding review finding prevents a case from passing.
