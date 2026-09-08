# Bagpiper

Bagpiper is a multimodal **SpeechLM** recipe. A Qwen3-8B(-Base) language model is
extended to both **understand** audio (speech / sound / music → text) and
**generate** audio (text → speech / sound / music), trained jointly with
text-only data so the model keeps its language ability while gaining audio I/O.

## Layout

This directory is the operational hub: data preparation, configs, launch
scripts, and `exp/` outputs. The model, dataloader, and trainer live in the
training library at [`espnet2/speechlm/`](../../../espnet2/speechlm) and are
invoked as `../../../espnet2/speechlm/bin/*.py`.

Unlike classic ESPnet recipes, Bagpiper does **not** use `run.sh` / `speechlm.sh`.
The entry points are the bespoke `launch_bagpiper_stage{1,2,3}_*.sh` scripts,
which call `torchrun ../../../espnet2/speechlm/bin/train.py` directly. The
inherited `speechlm.sh`, `steps/`, `utils/`, `pyscripts/`, `scripts/` symlinks are
mostly unused (only `utils/parse_options.sh`).

## Environment

The launch scripts source `./path.sh` → `conda activate dev`. Running the model
requires an environment with **flash-attention-3** (mandatory; the model asserts
it), **TorchTitan**, and **DeepSpeed**. Base models (Qwen3-8B-Base, the Xcodec
codec, and the Qwen3-Omni audio tower) are loaded from the Hugging Face cache, so
export `HF_HOME` and, for an offline node, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`;
export `PYTHONPATH` to the repo root when calling `bin/*.py` outside the launchers.

> **Registry gotcha** (`path.sh`): `ESPNET_DATASET_REGISTRY` is only exported when
> `hostname` matches the wavlab cluster (`dt*/gh*/gpu*`). On any other host,
> "registered" data specifiers resolve to nothing — set it manually if you need
> registered datasets off-cluster.

## Model

One unified multi-stream vocabulary: stream 0 carries text + special tokens,
streams 1+ carry audio codebooks. Configured under `multimodal_io:` in each train
YAML:

- **text** — Hugging Face `AutoTokenizer` (Qwen3-8B / Qwen3-8B-Base).
- **discrete_audio** — the audio *output* codec (Xcodec, 8 streams). Generation half of the vocab.
- **continuous_audio** — the audio *input* encoder (Qwen3-Omni audio tower), projected into
  embedding space by a linear adaptor. Understanding side.

The backbone is built by `ParallelHFModel`, which rebuilds `embed_tokens`/`lm_head`
to the unified vocab and copies the pretrained text embeddings into the text slice.

## Three-stage training

Each launcher uses ESPnet's `--stage/--stop_stage` convention with three internal
stages: **(1) length stats → (2) train → (3) inference**. Length stats are a **hard
prerequisite** for training (run stage 1 once per data mix + config).

| Stage | Launcher | Default config | Trains | Frozen |
|---|---|---|---|---|
| 1 — warmup   | `launch_bagpiper_stage1_warmup.sh`   | `conf/train_stage1_qwen3_base.yaml` | audio I/O adaptors | backbone + text emb |
| 2 — pretrain | `launch_bagpiper_stage2_pretrain.sh` | `conf/train_stage2_qwen3_base.yaml` | full backbone + I/O | codec / encoder |
| 3 — SFT      | `launch_bagpiper_stage3_sft.sh`      | `conf/train_stage3_qwen3_base.yaml` | backbone + text emb | codec / encoder |

```bash
# Stage 1 — length stats only (once per data mix + config)
./launch_bagpiper_stage2_pretrain.sh --stage 1 --stop_stage 1 --num_proc_per_node 8

# Stage 2 — distributed training (single node, 8 GPUs)
./launch_bagpiper_stage2_pretrain.sh --stage 2 --stop_stage 2 \
    --num_nodes 1 --num_proc_per_node 8 --node_rank 0 \
    --master_addr localhost --master_port 8888

# Stage 3 — sharded inference over the test specifiers
./launch_bagpiper_stage2_pretrain.sh --stage 3 --stop_stage 3
```

Stage 2 also has TorchTitan variants for scaling (`conf/train_stage2_qwen3_titan.yaml`,
`..._32b_titan.yaml`, `..._titan_moe.yaml`). The trainer backend is chosen from
`trainer.type` in the YAML: DeepSpeed ZeRO (default), TorchTitan FSDP2/HSDP, or
TorchTitan + pipeline parallel.

## Data preparation

```bash
local/data.sh --stage S --stop_stage S            # raw text/captions -> data_jsons/*.json
local/data_curation.sh --stage S --stop_stage S   # quality curation of registered audio datasets
../../../espnet2/speechlm/bin/prepare_dataset_json.py --triplets name,path,reader ...
```

Data mixes are named on the CLI with the specifier grammar
`task:name[:data_json][:factor]` (see `espnet2/speechlm/dataloader/iterator.py`).
Stage-3 SFT data is synthetic dialogues built by `local/sft_data_simulation/` (see
its `sft_data_simulation_doc.md`).

## Public checkpoint

The public checkpoint is the **stage-2 pretrained base at step 260000** (Qwen3-8B-Base
backbone), a DeepSpeed ZeRO checkpoint (~117 GB) from the Hugging Face dataset
`JinchuanTian/Bagpiper_checkpoints`. `conf/train_stage2_qwen3_base.yaml` reconstructs
the model behind it. For inference or initialization, point `--model-checkpoint` at the
model-weights file inside the checkpoint,
`.../step_260000/global_step259988/mp_rank_00_model_states.pt`.

## Inference

```bash
../../../espnet2/speechlm/bin/inference.py \
    --rank JOB --world-size N \
    --train-config exp/<run>/train.yaml \
    --inference-config conf/inference_pt.yaml \        # text decode (pretrain); inference_sft.yaml for text+audio
    --model-checkpoint <ckpt>/global_step259988/mp_rank_00_model_states.pt \
    --output-dir <out> --test-registered-specifier "<spec>" --num-worker 3
```

Inference is sharded by `--rank/--world-size` and each process forks `--num-worker`
model replicas; it writes `.wav` segments + `results.json`. `conf/inference_pt.yaml`
decodes text only (pretrain); `conf/inference_sft.yaml` decodes the `<think>` + caption
and then generates audio (SFT).
