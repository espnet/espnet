# Bagpiper SpeechLM recipe

Fine-tune Bagpiper-Base on the released audio generation and understanding dialogues. The ESPnet3 launcher runs `collect_stats`, `train` and `infer`. Both full-parameter and LoRA training use the same audio preprocessing, vocabulary, assistant-turn loss and native TorchTitan trainer.

## Data Preparation

Install the ESPnet3 training and SpeechLM extras with `python -m pip install -e ".[train,speechlm]"` from the repository root, then install the native extensions for your hardware. Activate the environment and run the commands below from `egs3/bagpiper/speechlm`. An installed ESPnet can also copy the recipe and its native SFT configuration into a standalone directory with `espnet3 clone bagpiper/speechlm --project my_bagpiper`. Run the same commands from `my_bagpiper` in that case. [SpeechLM installation](../../../espnet3/systems/speechlm/src/INSTALL.md)

```bash
export BAGPIPER_PREPARED="$PWD/data/sft/prepared"
```

Point `BAGPIPER_PREPARED` at your existing prepared data if available. The recipe reads `{generation,understanding}/{train,valid}.json` with their original dialogue IDs, audio paths and role order. The commands below prepare both partitions from the pinned release, which contains approximately 719 GB of compressed archives. To reproduce the measured subset, run the final preparation command twice, first with `--partition generation --max-shards 2` and then with `--partition understanding --max-shards 1`. Use the same data root in both commands. The helper assigns validation by audio hash. [SFT data release](https://huggingface.co/datasets/espnet/Bagpiper_SFT_Data_ESPnet)

```bash
python -m pip install zstandard
revision=643639bcd57d2cd4a66dbe4e081e2ff678da0ac4
hf download espnet/Bagpiper_SFT_Data_ESPnet prepare_espnet.py \
    --repo-type dataset --revision "$revision" --local-dir data/tools
python data/tools/prepare_espnet.py --repo-id espnet/Bagpiper_SFT_Data_ESPnet \
    --revision "$revision" --data-root "$PWD/data/sft" \
    --partition generation --partition understanding
```

## Commands

### Full-parameter supervised fine-tuning

The full-parameter preset uses eight H100 80 GB GPUs on one node, BF16, FlashAttention 3 and a 4,096-token pack per GPU. It trains the whole decoder, multimodal embeddings, continuous audio encoder and audio projection. The discrete Xcodec tokenizer stays fixed because training consumes its integer token IDs. The loader skips dialogues longer than the pack budget. To include longer dialogues in either mode, increase `data_loading.batch_size` in a copy of the native YAML and check peak memory with that setting. [Full training settings](conf/train_full.yaml)

Collect lengths once for both runs. Use a fresh `stats_dir` when changing the data or preprocessing because the collector reuses files by dataset name.

```bash
hf download espnet/bagpiper base.pt --revision cde214717b75190c0b117c0f3ad59b1412e3c695 \
    --local-dir models/bagpiper
export BAGPIPER_INIT="$PWD/models/bagpiper/base.pt"
export BAGPIPER_NGPU=8
python run.py --stages collect_stats train --training_config conf/training_full.yaml
```

The first run requires initial weights. On a retry, a complete distributed checkpoint in `exp/bagpiper_full/checkpoints` takes precedence over `BAGPIPER_INIT` and restores the optimizer, scheduler and step. Use a fresh `exp_dir` for a new experiment, and keep the original world size, gradient accumulation and data settings when resuming.

```bash
unset BAGPIPER_INIT
python run.py --stages train --training_config conf/training_full.yaml
```

For existing decoder-only experiments in `exp/bagpiper_sft`, continue to use `conf/training.yaml` with the original GPU count. That preset keeps the continuous audio encoder frozen.

### LoRA supervised fine-tuning

LoRA starts a separate experiment in `exp/bagpiper_lora` from the same released base checkpoint. It trains rank-16 adapters on the query, key, value and output projections in every decoder layer. The base weights stay frozen. The default uses one H100 80 GB, a 4,096-token pack and eight gradient accumulation steps. For two sharded workers, set `BAGPIPER_LORA_NGPU=2`. To keep the same 32,768 token slots per update, set `trainer.gradient_accumulation_steps` to 4 in a copy of the native YAML and point `BAGPIPER_LORA_CONFIG` at that file. [LoRA training settings](conf/train_lora.yaml)

```bash
export BAGPIPER_INIT="$PWD/models/bagpiper/base.pt"
export BAGPIPER_LORA_NGPU=1
python run.py --stages train --training_config conf/training_lora.yaml
```

To run LoRA first on a new dataset, include `collect_stats` before `train`. To start LoRA from your full fine-tune, point `BAGPIPER_INIT` at its exported `model.pt` and use a fresh LoRA experiment directory. To resume an interrupted LoRA run, repeat the original command and settings. Its distributed checkpoints include the base and adapter weights, optimizer, scheduler and step. Allow approximately 35 GiB per saved LoRA step or 101 GiB per full-training step on the tested configurations.

For a shorter experiment, copy the matching native YAML to a new file, set `trainer.max_step`, `trainer.save_interval`, and the warmup and decay steps under `trainer.lr_scheduler`. Point `BAGPIPER_TRAIN_CONFIG` (full) or `BAGPIPER_LORA_CONFIG` (LoRA) at the copy and use a fresh experiment directory in the ESPnet3 training config. Keep the order of `multimodal_io` because it determines the checkpoint's token IDs.

### Audio understanding and generation

To decode with your fine-tuned weights, export a completed step on a host with RAM for the model tensors. The exporter folds trained LoRA updates into the decoder weights and produces a standalone model for either training mode. Set `run_dir` to the experiment you trained and `step` to a saved step.

```bash
run_dir=exp/bagpiper_lora   # or exp/bagpiper_full
step=2500
export BAGPIPER_MODEL="$PWD/models/$(basename "$run_dir")-step-$step"
mkdir -p "$BAGPIPER_MODEL"
python -m espnet2.speechlm.bin.export_checkpoint \
    --checkpoint-dir "$run_dir/checkpoints/step_$step" \
    --output "$BAGPIPER_MODEL/model.pt" --dtype bfloat16
cp "$run_dir/train.yaml" "$BAGPIPER_MODEL/train.yaml"
hf download espnet/bagpiper-sft inference_text.yaml inference_audio.yaml \
    --revision 5646600302bf2670214c601721e44f6d36145c26 \
    --local-dir "$BAGPIPER_MODEL"
```

Both inference configs read the directory selected by `BAGPIPER_MODEL`. With that variable unset they load the published `espnet/bagpiper-sft`. Understanding uses the validation audio and question. Generation uses the validation text instruction. Both retain the system message when present and stop reading the conversation at the first assistant turn. The supplied configs use one worker and decode one request at a time. [Instruction-tuned model](https://huggingface.co/espnet/bagpiper-sft)

```bash
python run.py --stages infer --inference_config conf/inference.yaml
python run.py --stages infer --inference_config conf/inference_generation.yaml
```

Understanding writes `exp/bagpiper_sft/understanding/valid/text.scp`. Generation writes `exp/bagpiper_sft/generation/valid/{text,audio}.scp` and WAV files. Both commands recompute their outputs on each run. Choose a new `inference_dir` to keep results from two checkpoints side by side.

## Results

Both fine-tuning runs started from the same released Bagpiper-Base and completed one packed epoch on a prefix of the public SFT release. The first two generation shards and the first understanding shard yielded 2,032 generation and 1,013 understanding training dialogues. Validation used all 16 generation and 11 understanding dialogues assigned by the preparer's default audio-hash split. The reported cross-entropies are teacher-forced losses on assistant outputs. This experiment covers three shards of a 2,667,017-dialogue release. The shipped presets specify 50,000 updates. [Base checkpoint](https://huggingface.co/espnet/bagpiper/tree/cde214717b75190c0b117c0f3ad59b1412e3c695) · [SFT release](https://huggingface.co/datasets/espnet/Bagpiper_SFT_Data_ESPnet/tree/643639bcd57d2cd4a66dbe4e081e2ff678da0ac4)

Both runs used BF16, 4,096-token packs, 32,768 token slots per update, and two data workers per rank. AdamW warmed up for 19 updates and decayed through the last update. Median update times exclude the first two updates. Device memory was sampled every three seconds.

Both BF16 exports reloaded for audio understanding and generation through the ESPnet3 inference stage. On one held-out generation prompt, an H100 using the released decoding settings (audio temperature 0.8, guidance scale 3, seed 42) produced 11.82 seconds of 16 kHz audio with full fine-tuning and 4.02 seconds with LoRA. The respective RMS levels were −17.5 and −22.1 dBFS. The reference clip lasted 10.00 seconds at −19.0 dBFS RMS. A separate CPU check used BF16, SDPA and a 64-step greedy cap per segment to verify both inference modes.

### Full-parameter fine-tuning

Eight H100 80 GB GPUs trained 8,916,359,552 parameters for 191 updates, with one microbatch per GPU per update and peak learning rate 1e-5. Validation cross-entropy was **1.688 for generation** and **0.4438 for understanding**. Median update time was 0.940 s, and peak sampled memory was 38,823 MiB per GPU. [Full preset](conf/training_full.yaml)

### LoRA fine-tuning

Two H100 80 GB GPUs trained 15,335,424 adapter parameters for 189 updates, with four microbatches per GPU per update and peak learning rate 1e-4. Validation cross-entropy was **1.757 for generation** and **0.6490 for understanding**. Median update time was 2.651 s, and peak sampled memory was 30,635 MiB per GPU. Export merged all 144 trained attention projections. All 1,238 other exported tensors matched Bagpiper-Base byte for byte at the export dtype. [LoRA preset](conf/training_lora.yaml)

The single-H100 configuration completed three optimizer updates with eight microbatches per update, validated on the same 27 dialogues and saved a distributed checkpoint. Peak sampled memory was 53,937 MiB on one H100 80 GB.

### Released instruction-tuned model

Two held-out requests were also checked with the published Bagpiper-SFT checkpoint on CPU using BF16 and SDPA. With four greedy steps, the understanding adapter returned the same text as `LocalBagpiper.describe`. With a 24-step cap per segment, `run.py --stages infer` wrote a text record and 5,440 frames of 16 kHz audio for the generation request. [Released Bagpiper SFT model and results](https://huggingface.co/espnet/bagpiper-sft)
