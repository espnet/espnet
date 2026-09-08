# Bagpiper

Training recipes for [Bagpiper: Solving Open-Ended Audio Tasks via Rich Captions](https://openreview.net/forum?id=FuHs64E3X6),
an audio foundation model for understanding and generation through rich captions.

[Demo](https://bagpiper-cmu.github.io/) |
[Models and datasets](https://huggingface.co/collections/espnet/bagpiper) |
[Bagpiper-TTS recipe](../../bagpiper_tts/speechlm1/README.md)

## Setup and inputs

Activate an environment installed with the [SpeechLM installation guide](../../../espnet2/speechlm/INSTALL.md).
Until the integration PRs are merged, use this recipe together with installation
[#6645](https://github.com/espnet/espnet/pull/6645) and trainer
[#6646](https://github.com/espnet/espnet/pull/6646).

This recipe starts from prepared training and validation data:

- SpeechLM dataset JSON manifests containing `data_entry` and `samples`, with
  paths to the corresponding text, audio, or dialogue files. The Hub Parquet
  releases are not direct inputs to the trainer.
- Length statistics in `stats_<task>_<name>.jsonl`, with one
  `{"example_id": length}` record per line, for each dataset/task combination.

Data preparation is outside this recipe. Pass datasets as space-separated
`task:name:dataset.json[:factor]` specifiers. Supported tasks are `text_to_audio`,
`audio_to_text`, `text_only`, and `dialogue`; use distinct names for training and
validation splits. Registered datasets can instead use
`--train-registered-specifier` / `--valid-registered-specifier` with
`task:name[:factor]` and `ESPNET_DATASET_REGISTRY`.

The configurations use Qwen3-8B-Base, Xcodec, the Qwen3-Omni audio encoder, and
TorchTitan FSDP2. FlashAttention-3 is an example for Hopper GPUs such as H100;
on other GPUs, choose a supported backend for both `model.model_conf.attn_implementation`
and `multimodal_io.continuous_audio.attn_implementation`.

## Training

| Configuration | Initialization | Trainable components |
| --- | --- | --- |
| `conf/train_warmup.yaml` | Qwen3-8B-Base | Expanded token embeddings, output head, stream embeddings, audio adaptor, final norm |
| `conf/train_pretrain.yaml` | Warmup DCP checkpoint | Language model and multimodal embeddings/adaptor |
| `conf/train_sft.yaml` | Pretraining DCP checkpoint | Language model and multimodal embeddings/adaptor |

The audio encoder and codec stay frozen in every stage. Warmup also freezes
decoder layers; the text rows of the expanded embeddings remain trainable.
Token budgets and schedules are starting settings for this trainer, not an exact
reproduction of the original DeepSpeed training run. Adjust `data_loading.batch_size`
(token budget per GPU), `trainer.gradient_accumulation_steps`, `trainer.max_step`,
and `trainer.lr_scheduler` for your data and hardware. `dp_shard: -1` uses the
GPU count selected by the launcher.

Run from `egs2/bagpiper/speechlm1`. Replace the example data paths with prepared
manifests and statistics:

```bash
./run.sh --ngpu 8 \
    --train-config conf/train_warmup.yaml --output-dir exp/warmup \
    --stats-dir /path/to/pretrain_stats \
    --train-unregistered-specifier "text_to_audio:train:/path/to/train.json audio_to_text:train:/path/to/train.json" \
    --valid-unregistered-specifier "text_to_audio:valid:/path/to/valid.json audio_to_text:valid:/path/to/valid.json"
```

For pretraining, use the same data arguments with
`--train-config conf/train_pretrain.yaml --output-dir exp/pretrain` and
`--resume-path exp/warmup/checkpoints/step_10000`.

For SFT, use prepared dialogue manifests containing the instruction, rich-caption,
and audio turns for each task:

```bash
./run.sh --ngpu 8 \
    --train-config conf/train_sft.yaml --output-dir exp/sft \
    --resume-path exp/pretrain/checkpoints/step_600000 \
    --stats-dir /path/to/sft_stats \
    --train-unregistered-specifier "dialogue:sft_train:/path/to/sft_train.json" \
    --valid-unregistered-specifier "dialogue:sft_valid:/path/to/sft_valid.json"
```

An explicit `--resume-path` loads model weights from a PyTorch Distributed
Checkpoint (DCP) directory and starts a new optimizer, scheduler, and step counter;
use a new output directory when switching stages. The current Titan trainer cannot
initialize directly from the Hub's `base.pt` or `model.pt` files.

To resume an interrupted stage, rerun its command with the same data, configuration,
and output directory, **omitting `--resume-path`**. The latest checkpoint in
`output_dir/checkpoints/step_*` restores the model, optimizer, scheduler, and step.
The launcher saves batch assignments by default (`--save-loader-state true`).

For multiple nodes, run the same command on each node with `--num-nodes N`,
`--node-rank R`, `--master-addr HOST`, and `--master-port PORT`; use the same
data and a shared output directory. `--ngpu` is the number of GPUs per node.
All relative paths are resolved from the recipe directory. See `./run.sh --help`
for options; W&B is disabled by default.
