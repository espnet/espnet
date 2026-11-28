# Bagpiper SpeechLM recipe

`SpeechLMSystem` provides `collect_stats`, `train` and `infer`. SpeechLM's model factory and trainer already construct the model and optimize it from the native YAML, so this port needs no ESPnet2 Task class. The training configuration below selects native full-decoder SFT. LoRA training is a separate milestone. [ESPnet2 migration guide](https://espnet.github.io/espnet/espnet3/guides/migrating-from-espnet2/) · [ESPnet3 recipe conventions](https://github.com/espnet/espnet/blob/master/.agents/egs3/AGENTS.md)

## Data Preparation

Install the ESPnet3 training and SpeechLM extras with `python -m pip install -e ".[train,speechlm]"` from the repository root, then install the native extensions for your hardware. Activate the environment and run the commands below from `egs3/bagpiper/speechlm`. An installed ESPnet can also copy the recipe and its native SFT configuration into a standalone directory with `espnet3 clone bagpiper/speechlm --project my_bagpiper`. Run the same commands from `my_bagpiper` in that case. [SpeechLM installation](../../../espnet3/systems/speechlm/src/INSTALL.md)

```bash
export BAGPIPER_PREPARED="$PWD/data/sft/prepared"
```

Point `BAGPIPER_PREPARED` at your existing prepared data if available. The recipe reads `{generation,understanding}/{train,valid}.json` with their original dialogue IDs, audio paths and role order. To prepare a small public prefix on the current machine, run the pinned release helper. Omit `--max-shards 2` to prepare both full partitions. [SFT data release](https://huggingface.co/datasets/espnet/Bagpiper_SFT_Data_ESPnet)

```bash
revision=643639bcd57d2cd4a66dbe4e081e2ff678da0ac4
hf download espnet/Bagpiper_SFT_Data_ESPnet prepare_espnet.py \
    --repo-type dataset --revision "$revision" --local-dir data/tools
python data/tools/prepare_espnet.py --repo-id espnet/Bagpiper_SFT_Data_ESPnet \
    --revision "$revision" --data-root "$PWD/data/sft" \
    --partition generation --partition understanding --max-shards 2
```

## Commands

### Supervised fine-tuning

The native configuration uses BF16, FlashAttention 3 and an 8,192-token pack per GPU. The target for full-decoder SFT is two NVIDIA H100 80 GB GPUs on one node. `BAGPIPER_NGPU` sets the number of local TorchTitan workers. Peak training memory has yet to be measured on that setup. For a smaller run, set `BAGPIPER_TRAIN_CONFIG` to a copy of `conf/train_sft.yaml` and adjust the token budget, step count and save interval. Preserve the order of `multimodal_io` in that copy because it determines the checkpoint's token IDs. [Native SFT config](conf/train_sft.yaml)

Statistics use CPU workers and initialize the configured tokenizers and audio components. Use a fresh `stats_dir` when the data or its preprocessing changes. The native collector reuses files by dataset name. Use the same native configuration and statistics for training.

```bash
hf download espnet/bagpiper base.pt --revision cde214717b75190c0b117c0f3ad59b1412e3c695 \
    --local-dir models/bagpiper
export BAGPIPER_INIT="$PWD/models/bagpiper/base.pt"
export BAGPIPER_NGPU=2
python run.py --stages collect_stats train --training_config conf/training.yaml
```

The first run requires initial weights. On a retry, a complete distributed checkpoint in `exp/bagpiper_sft/checkpoints` takes precedence over `BAGPIPER_INIT` and restores the optimizer, scheduler and step. Use a fresh `exp_dir` for a new experiment, and keep the original world size, gradient accumulation and data settings when resuming.

```bash
unset BAGPIPER_INIT
python run.py --stages train --training_config conf/training.yaml
```

### Audio understanding and generation

Inference loads one model per worker and handles one request at a time. The supplied configuration uses one worker. The default model is the released instruction-tuned `espnet/bagpiper-sft`. Set `BAGPIPER_MODEL` to a local release directory to run from local files. The understanding config selects the validation audio and question, and the generation config selects the validation text instruction. Both retain the system message when present and stop reading the conversation at the first assistant turn. [Instruction-tuned model](https://huggingface.co/espnet/bagpiper-sft)

```bash
python run.py --stages infer --inference_config conf/inference.yaml
python run.py --stages infer --inference_config conf/inference_generation.yaml
```

Understanding writes `exp/bagpiper_sft/understanding/valid/text.scp`. Generation writes `exp/bagpiper_sft/generation/valid/{text,audio}.scp` and WAV files. Both commands recompute their outputs on each run. Choose a new `inference_dir` to keep results from two checkpoints side by side.

For a locally trained model, export an available training step into a new directory. Place its matching native training YAML and the release's two decoding configs beside the exported weights, then rerun the inference commands with `BAGPIPER_MODEL` set.

```bash
export BAGPIPER_MODEL="$PWD/models/bagpiper-local-sft"
mkdir -p "$BAGPIPER_MODEL"
python -m espnet2.speechlm.bin.export_checkpoint \
    --checkpoint-dir exp/bagpiper_sft/checkpoints/step_2500 \
    --output "$BAGPIPER_MODEL/model.pt" --dtype bfloat16
cp exp/bagpiper_sft/train.yaml "$BAGPIPER_MODEL/train.yaml"
hf download espnet/bagpiper-sft inference_text.yaml inference_audio.yaml \
    --local-dir "$BAGPIPER_MODEL"
```

## Results

### General Bagpiper SFT

`conf/training.yaml` initializes from Bagpiper-Base. `conf/inference.yaml` and `conf/inference_generation.yaml` use the checkpoint's native decoding settings. [Released Bagpiper SFT model and results](https://huggingface.co/espnet/bagpiper-sft)

Two held-out requests were checked with the released SFT checkpoint on CPU using BF16 and SDPA. With four greedy steps, the understanding adapter returned the same text as `LocalBagpiper.describe`. With a 24-step cap per segment, `run.py --stages infer` wrote a text record and 5,440 frames of 16 kHz audio for the generation request. ESPnet3 quality, H100 memory use and training throughput remain unmeasured.
