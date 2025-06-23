# Audioverse Benchmark Runner

A comprehensive benchmarking tool for audio tasks including captioning, classification, detection, and music analysis.
This runner runs multiple audio recipes in ESPnet with configurable arguments per recipe or common arguments to all recipes. It also supports both sequential and parallel execution of recipes.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Basic Usage](#basic-usage)
- [Common Workflows](#common-workflows)
- [Configuration System](#configuration-system)
- [Output Structure](#output-structure)
- [Available Tasks](#available-tasks)
- [Common Pitfalls](#common-pitfalls)
- [Troubleshooting](#troubleshooting)

## Overview

The Audioverse Benchmark Runner supports the following tasks:

- **Audio Captioning**: AudioCaps, Clotho
- **Audio-Text Reasoning**: Audio Entailment and Question Answering
- **General Sound Classification**: AudioSet (2M/20K), FSD50K, ESC-50
- **Bioacoustic Detection/Classification**: BEANS benchmark
- **Music Analysis**: GTZAN genre classification, NSynth instrument/pitch

## Quick Start

```bash
# Run a single task with openbeats
./run.sh --config_prefix openbeats_large --recipe esc50_f1

# Run multiple tasks
./run.sh --config_prefix openbeats_large --recipe "esc50_f1,esc50_f2,as20k"

# Run all tasks except specific ones
./run.sh --config_prefix openbeats_large --filter_recipe "beans,nsynth"

# Run with custom checkpoint path
# The yaml template must have BEATS_CHECKPOINT_PATH placeholder
./run.sh --config_prefix openbeats_large --recipe audiocaps_aac \
  --template_args "BEATS_CHECKPOINT_PATH:/path/to/your/different_checkpoint.pth"
```

## Prerequisites

1. **Environment Setup**: Needs ESPnet and transformers.
1. **Data**: Some recipes like audioset require you to download data and set the path in `db.sh`. Note: If `db.sh` contains `downloads` then this runner will automatically download the data.
1. **Templates**: Create configuration template for your model in `conf/template/{your_awesome_model}/`. You may use the ones for OpenBEATs as example.

The [ESPnet documentation about recipes](https://espnet.github.io/espnet/espnet2_tutorial.html#understanding-espnet2-recipes) is helpful for understanding how this runner fits into ESPnet recipe structure.

## Basic Usage

### Command Structure
```bash
./run.sh --config_prefix TEMPLATE_DIRECTORY [OPTIONS] [ADDITIONAL_RECIPE_ARGS]
```

### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--config_prefix` | **Required**. Template directory name. We store all recipe configs of a model in this directory | `--config_prefix openbeats_large` |
| `--recipe` | Specific recipes to run (comma-separated) | `--recipe "audiocaps_aac,clotho_aac"` |
| `--filter_recipe` | Exclude specific recipes | `--filter_recipe "beans,nsynth"` |
| `--run_name` | Name for this run (default: timestamp) | `--run_name "my_experiment"` |
| `--parallel` | Run recipes in parallel | `--parallel true` |
| `--dry_run` | Print commands without executing | `--dry_run true` |

### Arguments and Logging Options

| Option | Description | Example |
|--------|-------------|---------|
| `--template_args` | Replace template placeholders | `--template_args "CHECKPOINT_PATH:/path/to/awesome_model"` |
| `--recipe_args` | Arguments for recipe runners, appended at end to overwrite other default args in recipe | `--recipe_args "--stage 4 --batch_bins 800000"` |
| `--task_args` | Task-specific arguments. These are fed as `cls_args` or `asr1_args` depending on recipe. | `--task_args "--lr 0.001 --epochs 50"` |
| `--log_wandb` | Enable Weights & Biases logging. Please provide `wandb_entity` and `wandb_project` when using this. | `--log_wandb true` |

## Common Workflows

### 1. Evaluate Single Checkpoint on Multiple Tasks

For comparing a pre-trained model across different benchmarks:

```bash
./run.sh \
  --config_prefix openbeats_large \
  --recipe "audiocaps_aac,clotho_aac,cle_bert,aqa_yn_bert" \
  --template_args "MODEL_CHECKPOINT:/path/to/openbeats.pth,BATCH_SIZE:32" \
  --run_name "openbeats_evals"
```

### 2. Hyperparameter Sweep with Template Arguments

Run the same task with different configurations:

```bash
# Different learning rates
for lr in 0.001 0.0001 0.00001; do
  ./run.sh \
    --config_prefix openbeats_large \
    --recipe audiocaps_aac \
    --template_args "LEARNING_RATE:$lr,BATCH_SIZE:16" \
    --run_name "lr_sweep_${lr}" \
    --log_wandb true \
    --task_args "--wandb_entity your_team --wandb_project audio_benchmark"
done
```

### 3. Run All Tasks with Common Arguments

Apply common settings across all tasks:

```bash
./run.sh \
  --config_prefix openbeats_large \
  --recipe_args "--stage 4 --batch_bins 400000" \
  --log_wandb true \
  --task_args "--wandb_entity your_team --wandb_project audio_benchmark"
```

### 4. Parallel Execution

Run independent tasks simultaneously:

```bash
# Run ESC-50 cross-validation folds in parallel
./run.sh \
  --config_prefix openbeats_large \
  --recipe "esc50_f1,esc50_f2,esc50_f3,esc50_f4,esc50_f5" \
  --parallel true \
  --wait_time 10 \
  --run_name "esc50_crossval"
```

### 5. Focused Task Category Evaluation

Run specific task categories:

```bash
# Only bioacoustic tasks
./run.sh \
  --config_prefix openbeats_large \
  --recipe "beans_dcase,beans_enabirds,beans_gibbons,beans_hiceas,beans_rfcx" \
  --run_name "bioacoustic_benchmark"

# Only music tasks
./run.sh \
  --config_prefix openbeats_large \
  --recipe "gtzan,nsynth_instrument,nsynth_pitch" \
  --run_name "music_benchmark"
```

## Configuration System

### Template Structure

Create template recipe configs in `conf/template/{config_prefix}/{recipe_name}.yaml`:

```
conf/template/
├── openbeats_large/
│   ├── audiocaps_aac.yaml
│   ├── clotho_aac.yaml
│   └── esc50_f1.yaml
├── openbeats_base/
│   ├── audiocaps_aac.yaml
│   └── cle_bert.yaml
└── beats/
    └── audiocaps_aac.yaml
```

### Template Placeholders

Use ALL_CAPS placeholders in your templates:

```yaml
model_conf:
  checkpoint_path: CHECKPOINT_PATH

scheduler_conf:
    max_lr: MAX_LR
    min_lr: 5.0e-6
```

And then replace them dynamically with `--template_args`:
```bash
--template_args "CHECKPOINT_PATH:/path/to/model.pth,MAX_LR:0.001"
```

## Output Structure

```
audioverse/v1/
├── exp/
│   ├── runs/
│   │   └── {run_name}/
│   │       ├── conf/                    # Generated configs
│   │       │   └── {config_prefix}/
│   │       │       └── {recipe}.yaml
│   │       ├── {recipe1}.log           # Individual task logs
│   │       ├── {recipe2}.log
│   │       └── summary.txt             # Overall run summary
│   ├── {recipe1}/                      # Task-specific results
│   │   ├── [cls|asr]_{run_name}/
│   └── {recipe2}/
├── dump/                               # Processed data (if store_locally=true)
│   ├── {recipe1}/
│   └── {recipe2}/
├── data/                              # Raw data (if store_locally=true)
│   ├── {recipe1}/
│   └── {recipe2}/
└── logs/                              # Additional logging
```

### Key Output Files

- **`exp/runs/{run_name}/summary.txt`**: High-level success/failure summary
- **`exp/runs/{run_name}/{recipe}.log`**: Detailed logs for each task
- **`exp/{recipe}/[cls|asr]_{run_name}/`**: Training checkpoints and evaluation results in the same style as ESPnet

## Available Tasks

### Audio Captioning
- `audiocaps_aac`: AudioCaps pre-training
- `clotho_aac`: Clotho fine-tuning

### Audio-Text Classification
- `cle_bert`, `cle_clap`: Clotho Entailment
- `aqa_yn_bert`, `aqa_yn_clap`: Audio Question Answering (Yes/No)
- `aqa_open_bert`, `aqa_open_clap`: Audio Question Answering (Open)

### Sound Classification
- `audioset2m`: AudioSet 2M multi-label classification
- `audioset20k`: AudioSet 20K balanced subset multi-label classification
- `fsd50k`: Freesound Dataset 50K multi-label classification
- `esc50_f1` to `esc50_f5`: ESC-50 multi-class classification

### Bioacoustic Tasks (BEANS)
- **Detection**: `beans_dcase`, `beans_enabirds`, `beans_gibbons`, `beans_hiceas`, `beans_rfcx`
- **Classification**: `beans_watkins`, `beans_bats`, `beans_cbi`, `beans_humbugdb`, `beans_dogs`

### Music Tasks
- `gtzan`: Genre classification
- `nsynth_instrument`: Instrument recognition
- `nsynth_pitch`: Pitch classification

## Common Pitfalls

### 1. Missing `--config_prefix`
**Error**: "Error: --config_prefix is required"
**Solution**: Always specify a config prefix that matches your template directory
```bash
# ❌ Wrong
./run.sh --recipe audiocaps_aac

# ✅ Correct
./run.sh --config_prefix openbeats_large --recipe audiocaps_aac
```

### 2. Template Not Found
**Error**: "Template not found: conf/template/openbeats_large/audiocaps_aac.yaml"
**Solution**: Ensure template files exist for your config_prefix and recipes
```bash
# Check if template exists
ls conf/template/openbeats_large/audiocaps_aac.yaml
```

### 3. WandB Configuration Issues
**Error**: "--task_args must contain --wandb_entity if --log_wandb is true"
**Solution**: Provide wandb entity when enabling logging
```bash
# ✅ Correct
./run.sh --config_prefix basic --recipe audiocaps_aac \
  --log_wandb true --task_args "--wandb_entity your_team"
```

### 4. Parallel Execution Memory Issues
**Problem**: GPU out of memory with parallel execution
**Solutions**:
- Reduce batch sizes in templates
- Limit number of parallel recipes
- Run sequentially

```bash
# Safer parallel execution
./run.sh --config_prefix openbeats_large --recipe "task1,task2,task3" \
  --parallel false \
  --template_args "BATCH_SIZE:8"
```

## Troubleshooting

### Check Recipe Status
```bash
# View summary of latest run
cat exp/runs/{run_name}/summary.txt

# Check specific task log
tail -f exp/runs/{run_name}/{recipe}.log

# Check for common errors
grep -i "error\|failed\|fatal" exp/runs/{run_name}/*.log
```

### Validate Templates
```bash
# Test template generation
./run.sh --config_prefix basic --recipe audiocaps_aac --dry_run true

# Check generated config
cat exp/{run_name}/conf/basic/audiocaps_aac.yaml
```

### Debug Failed Tasks
1. Check the specific task log file
2. Verify template variable substitution
3. Ensure required data/checkpoints exist
4. Check disk space and memory usage
5. Validate recipe runner script exists and is executable

---

## Help and Support

For detailed command options:
```bash
./run.sh --help
```

Example configurations and templates can be found in the `conf/template/` directory.
