# Bagpiper-TTS

Supervised fine-tuning for Bagpiper-TTS, a speech synthesis model controlled by
natural-language instructions. Training dialogues contain a text request followed
by an assistant plan, rich caption, and target audio.

[Models and datasets](https://huggingface.co/collections/espnet/bagpiper) |
[SFT data](https://huggingface.co/datasets/espnet/Bagpiper_TTS_SFT_Data) |
[Bagpiper pretraining recipe](../../bagpiper/speechlm1/README.md)

Use the environment and prepared SpeechLM dataset manifests/length statistics
described in the [Bagpiper recipe](../../bagpiper/speechlm1/README.md#setup-and-inputs).
This recipe contains training only. The configuration uses Qwen3-8B-Base, Xcodec,
the Qwen3-Omni audio encoder, and TorchTitan; the encoder and codec stay frozen.
The example uses FlashAttention-3 for Hopper GPUs such as H100. Adjust both
attention backends, the per-GPU token budget, and the training schedule for your
hardware and data. These are starting settings, not the original run's step schedule.

Run from `egs2/bagpiper_tts/speechlm1`, using prepared `dialogue` datasets:

```bash
./run.sh --ngpu 8 \
    --train-config conf/train_sft.yaml --output-dir exp/sft \
    --resume-path ../../bagpiper/speechlm1/exp/pretrain/checkpoints/step_600000 \
    --stats-dir /path/to/tts_stats \
    --train-unregistered-specifier "dialogue:tts_train:/path/to/tts_train.json" \
    --valid-unregistered-specifier "dialogue:tts_valid:/path/to/tts_valid.json"
```

`--resume-path` must point to a Bagpiper pretraining DCP directory. It initializes
model weights and starts a fresh optimizer, scheduler, and step counter in the
new output directory. The current Titan trainer cannot load the Hub's `base.pt`
or `model.pt` files for training initialization.

To continue an interrupted SFT run, repeat the command with the same data,
configuration, and output directory, omitting `--resume-path` to restore the model,
optimizer, scheduler, and step from the latest checkpoint. `./run.sh --help` lists
shared launcher options, including multi-node training and W&B logging.
