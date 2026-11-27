# SpeechLM recipe template

`SpeechLMSystem` runs `collect_stats` and `train` using the native SpeechLM configuration, sampler and trainer. Set `speechlm_config` to the model YAML and declare the train and validation data specifiers in a training config. The `infer` stage uses the ESPnet3 runner and a modality-specific input/output contract.

The Bagpiper recipe includes working understanding and generation configs, the prepared dialogue format, and checkpoint instructions. [Bagpiper recipe](../../bagpiper/speechlm/README.md)
