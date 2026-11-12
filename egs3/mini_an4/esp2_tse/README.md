# Mini AN4 TSE smoke recipe

This fixture creates deterministic two-source mixtures from the small AN4 archive,
trains a tiny TD-SpeakerBeam for one batch, and checks inference, scoring and model/demo
packing. It is a software integration check, not a separation benchmark.

Clone with `espnet3 clone mini_an4/esp2_tse --project /tmp/tse-smoke` and copy
`egs3/mini_an4/esp2_asr/downloads.tar.gz` into that directory. From the clone, run:

```bash
python run.py --stages create_dataset collect_stats train infer measure pack_model pack_demo \
  --training_config conf/training.yaml --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml --publication_config conf/publication.yaml \
  --demo_config conf/demo.yaml
```

`ci/test_integration_espnet3_tse.sh` runs this flow in a temporary clone and checks
packed/demo/native output parity from outside the recipe. No model is uploaded.
