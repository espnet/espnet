# Bagpiper

Training recipes for [Bagpiper: Solving Open-Ended Audio Tasks via Rich Captions](https://openreview.net/forum?id=FuHs64E3X6),
an audio foundation model for understanding and generation through rich captions.

[Demo](https://bagpiper-cmu.github.io/) |
[Models and datasets](https://huggingface.co/collections/espnet/bagpiper) |
[Bagpiper-TTS recipe](../../bagpiper_tts/speechlm1/README.md)

## Setup and inputs

Activate an environment installed with the [SpeechLM installation guide](../../../espnet2/speechlm/INSTALL.md).
The launcher uses the active Python environment and invokes `torchrun` directly.
See the [template's training-only layout](../../TEMPLATE/speechlm1/README.md#training-only-recipes)
for environment activation, scheduler usage, and differences from `speechlm.sh`.

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

| Stage and configuration | Purpose | Initialization |
| --- | --- | --- |
| Warmup: `conf/train.yaml` (default) | Align expanded embeddings and the audio adaptor with the frozen decoder | Qwen3-8B-Base |
| Pretraining: `conf/tuning/train_pretrain.yaml` | Learn the joint audio/text model | Warmup DCP checkpoint |
| SFT: `conf/tuning/train_sft.yaml` | Specialize the model for instruction dialogues | Pretraining DCP checkpoint |

The audio encoder and codec stay frozen in every stage. Warmup also freezes
decoder layers; expanded token embeddings (including text rows), the output head,
stream embeddings, the audio adaptor, and final norm remain trainable. Pretraining
and SFT also train the decoder.

| Stage | `max_step` | Peak learning rate | LR warmup steps | Gradient accumulation | `min_lr_ratio` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warmup | 10,000 | 5e-4 | 100 | 4 | 1.0 |
| Pretraining | 600,000 | 1e-4 | 5,000 | 4 | 0.3 |
| SFT | 50,000 | 1e-5 | 1,000 | 1 | 0.1 |

Warmup keeps the learning rate constant after its first 100 steps
(`min_lr_ratio: 1.0`). The other stages use cosine decay after LR warmup.
All stages use a packed-token budget of 8,192 per GPU per micro-batch. At the same
GPU count, SFT's accumulation of 1 gives one quarter of the effective batch budget
of warmup/pretraining, which accumulate 4 micro-batches.

These schedules are starting settings for this trainer, not a reproduction of the
original DeepSpeed run. With 8 GPUs, the pretraining budget is approximately
`8192 × 4 × 8 = 262,144` tokens per optimizer step, or 157.3 billion token slots
over 600,000 steps; actual packed lengths vary. This recipe has no measured runtime
yet. After measuring `time/iter` in seconds, estimate training GPU-days as
`max_step × seconds_per_step × GPU_count / 86400`, plus validation and checkpoint
overhead. Adjust the token budget, accumulation, `max_step`, and LR schedule for
your run. `dp_shard: -1` uses the GPU count selected by the launcher.

Run from `egs2/bagpiper/speechlm1`. Replace the example data paths with prepared
manifests and statistics:

```bash
./run.sh --ngpu 8 \
    --stats-dir /path/to/pretrain_stats \
    --train-unregistered-specifier "text_to_audio:train:/path/to/train.json audio_to_text:train:/path/to/train.json" \
    --valid-unregistered-specifier "text_to_audio:valid:/path/to/valid.json audio_to_text:valid:/path/to/valid.json"
```

For pretraining, use the same data arguments with
`--train-config conf/tuning/train_pretrain.yaml --output-dir exp/pretrain` and
`--resume-path exp/warmup/checkpoints/step_10000`.
The example checkpoints `step_10000` and `step_600000` are the final checkpoints
of the default warmup and pretraining schedules; change them if you change the
schedule or select an earlier checkpoint.

For SFT, use prepared dialogue manifests containing the instruction, rich-caption,
and audio turns for each task:

```bash
./run.sh --ngpu 8 \
    --train-config conf/tuning/train_sft.yaml --output-dir exp/sft \
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
