# LibriSpeech 100h ESPnet3 recipe

This recipe follows the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) layout and uses the
latest `espnet3` package utilities.

## Quick start

```bash
# 1) Convert LibriSpeech to Hugging Face format (run once)
python run.py --stage create_dataset --input_dir /path/to/LibriSpeech --output_dir data

# 2) Train with the default Branchformer configuration
python run.py --stage train --train_tokenizer --collect_stats

# 3) Decode and score
python run.py --stage evaluate
```

All hyper-parameters are stored in `configs/` and can be overridden using Hydra syntax. For example:

```bash
python run.py --stage train --train_overrides trainer.max_epochs=10 runtime.device=cuda
```

To debug a single decoding sample:

```bash
python run.py --stage evaluate --debug_sample --eval_overrides runtime.debug_test=test-other
```
