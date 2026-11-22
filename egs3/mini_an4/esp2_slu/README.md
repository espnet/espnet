# mini_an4 SLU demo

The integration fixture for the `esp2_slu` system, used by
`ci/test_integration_espnet3.sh`. It is not a real SLU task: AN4's speaker id
stands in for a SLURP `scenario_action` label, so the target text is
`"<intent> <transcript>"` and the system, its inference runner and
`IntentAccuracy` all run over it.

The model is the smallest thing that still exercises `SLUTask`: no BERT
post-decoder and no deliberation encoder, so nothing is downloaded. The
two-pass transcript input that `egs3/slurp/esp2_slu` uses is therefore not
covered here; `test/espnet3/systems/esp2_slu/` covers it in unit tests.

```bash
. path.sh
python run.py \
    --stages create_dataset train_tokenizer collect_stats train infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```
