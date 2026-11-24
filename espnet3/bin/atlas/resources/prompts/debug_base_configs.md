## Base configs
Runtime keys are injected automatically — do NOT include them.


### training config
```yaml
${training_config}```

### Inference config (patch it only to repair an inference-stage failure)
```yaml
${inference_config}```

### Fixed metrics config (immutable; do not propose patches)
```yaml
${metrics_config}```
