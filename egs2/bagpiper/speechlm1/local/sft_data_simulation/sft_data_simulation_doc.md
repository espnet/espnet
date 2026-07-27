# SFT Data Simulation Pipeline

This pipeline generates SFT (Supervised Fine-Tuning) training data for open-ended text-to-audio generation. It creates synthetic user requests and reasoning traces from existing rich audio captions.

## Overview

The pipeline transforms pretrain data (rich caption + audio) into SFT dialogue format:
```
User Request -> <think>Reasoning Trace</think> + Rich Caption -> Audio
```

## Prerequisites

1. **vLLM Service**: The pipeline requires running vLLM service(s).
   - Launch script: `/work/nvme/bbjs/jtian1/tools/vllm/launch_vllm_qwen3_235b.sh`
   - You can use different services/models for Stage 1, Stage 2, and Stage 4
   - **Multi-instance support**: Use `:` to separate multiple URLs serving the same model for load balancing

2. **Input Data**: Rich caption metadata in JSONL format with fields:
   - `qwen_caption`: The rich audio description
   - `audio_path`: Path to the audio file
   - `dataset`: Dataset name

## Usage

### Basic Usage

```bash
# Run the full pipeline
./local/sft_data_simulation.sh --version v1

# With custom vLLM URLs and models per stage
./local/sft_data_simulation.sh \
    --version v1 \
    --vllm_url_stage1 http://host1:8000/v1 \
    --vllm_url_stage2 http://host2:8000/v1 \
    --vllm_url_stage4 http://host3:8000/v1 \
    --model_stage1 "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8" \
    --model_stage2 "Qwen/Qwen3-8B" \
    --model_stage4 "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

# With multiple vLLM instances per stage (load balancing)
# Use ":" to separate multiple URLs serving the same model
./local/sft_data_simulation.sh \
    --version v1 \
    --vllm_url_stage1 "http://host1:8000/v1:http://host2:8000/v1:http://host3:8000/v1" \
    --vllm_url_stage2 "http://host4:8000/v1:http://host5:8000/v1" \
    --model_stage1 "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8" \
    --model_stage2 "Qwen/Qwen3-8B"
```

### Debug Mode

```bash
# Process only 100 samples for testing
./local/sft_data_simulation.sh --version debug_v1 --num_samples 100
```

### Run Specific Stages

```bash
# Only Stage 1: Generate user requests
./local/sft_data_simulation.sh --version v1 --stage 1 --stop_stage 1

# Only Stage 2: Generate reasoning traces
./local/sft_data_simulation.sh --version v1 --stage 2 --stop_stage 2

# Only Stage 3: Assemble dialogues
./local/sft_data_simulation.sh --version v1 --stage 3 --stop_stage 3

# Only Stage 4: Quality validation
./local/sft_data_simulation.sh --version v1 --stage 4 --stop_stage 4
```

### Resume Interrupted Job

```bash
# Resume from where it stopped
./local/sft_data_simulation.sh --version v1 --resume true
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--version` | (required) | Version tag for output subfolder |
| `--stage` | 1 | Start stage |
| `--stop_stage` | 100 | Stop stage |
| `--vllm_url_stage1` | http://localhost:8000/v1 | vLLM URL(s) for user request generation (use `:` to separate multiple) |
| `--vllm_url_stage2` | http://localhost:8000/v1 | vLLM URL(s) for reasoning trace generation (use `:` to separate multiple) |
| `--vllm_url_stage4` | http://localhost:8000/v1 | vLLM URL(s) for quality judge (use `:` to separate multiple) |
| `--model_stage1` | Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 | Model for user request generation |
| `--model_stage2` | Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 | Model for reasoning trace generation |
| `--model_stage4` | Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 | Model for quality judge |
| `--input_file` | .../part2_pretrain_curation/metadata.jsonl | Input JSONL file |
| `--output_base` | .../sft_data/sft_simulation | Base output directory |
| `--num_workers` | 256 | Number of concurrent API requests |
| `--timeout` | 1200 | Request timeout in seconds |
| `--num_samples` | -1 | Number of samples to process (-1 for all) |
| `--resume` | false | Resume from existing progress |

## Pipeline Stages

### Stage 1: Generate User Requests

Generates 2 user requests per sample at different styles:
- **realistic**: Directly describes the audio events, sounds, and details
- **imaginary**: Describes a scene/feeling/atmosphere/concept (NOT audio directly). The audio is inferred from the description.

**Features:**
- **LLM-assisted persona selection**: For each sample, the LLM first selects a compatible persona based on the audio content (e.g., won't assign "voice-over artist" to non-speech audio). This ensures semantic coherence between persona and audio type.
- **Realistic requests**: Directly describe what audio the user wants - specific sounds, music, effects, etc.
- **Imaginary requests**: Abstract/creative prompts that describe scenes, feelings, or atmospheres rather than specific audio. Example: "Walking through an ancient forest at dawn, mist rising from the ground" instead of "forest ambient sounds".
- **Length**: 15-80 words (excluding quoted speech/lyrics). Length varies naturally regardless of audio complexity for diversity.
- **Human-like**: Written as real humans would type - natural language, avoids technical audio jargon (e.g., 'frequency response', 'sidechain', 'EQ').
- **Transcription preservation**: If the caption contains spoken words, lyrics, or transcribed text, the exact text is preserved in BOTH request types. Quoted text does NOT count toward word limit.
- **Shuffled sampling**: When `--num_samples` is specified, samples are shuffled (seed=42) before selection for diversity.

**Note**: Each sample requires 2 LLM calls (persona selection + request generation).

**Output**: `stage1_user_requests/user_requests.jsonl`

### Stage 2: Generate Reasoning Traces

Generates structured reasoning traces that bridge user requests to rich captions:
1. **User Intent**: What the user explicitly requested
2. **Inferred Details**: Reasonable details to fill gaps
3. **Quality Considerations**: What makes audio pleasant/professional
4. **Generation Plan**: How to create the audio

Runs separately for each detail level (realistic/imaginary).

**Output**: `stage2_reasoning_traces/reasoning_{level}.jsonl`

### Stage 3: Assemble Dialogues

Combines all components into the final dialogue format:
```json
{
  "example_id": "dataset_idx_level",
  "messages": [
    ["system", "text", "You are a helpful assistant..."],
    ["user", "text", "user request..."],
    ["assistant", "text", "<think>\n1. User Intent:...\n</think>\n\nRich caption..."],
    ["assistant", "audio", "/path/to/audio.flac"]
  ],
  "metadata": {
    "dataset": "...",
    "detail_level": "realistic|imaginary",
    "persona": "a content creator making YouTube videos",
    "original_idx": 123
  }
}
```

**Output**: `stage3_dialogues/dialogues_{level}.jsonl`

### Stage 4: LLM-as-Judge Quality Validation

Evaluates the quality of the **whole training example** using LLM-as-judge. Focuses on whether the example would be good training data.

**Category 1: User Request Quality (5 dimensions):**
- `naturalness`: Does it sound like a real human wrote it?
- `clarity`: Is the intent clear and unambiguous?
- `specificity`: Appropriate detail level? (not too vague, not overly prescriptive)
- `feasibility`: Is this a reasonable request an audio system could fulfill?
- `language_quality`: Avoids technical audio jargon? (no "sidechain", "EQ", etc.)

**Category 2: Request-Response Alignment (5 dimensions):**
- `intent_match`: Does the caption address what the user wanted?
- `content_coverage`: Are key elements from request reflected in caption?
- `no_contradictions`: Does the caption avoid contradicting the request?
- `scope_match`: Is response detail appropriate for request detail?
- `transcription_match`: If speech/lyrics exist, do they EXACTLY match in both? (Score 5 if no transcription - non-speech audio passes automatically)

**Category 3: Rich Caption Quality (4 dimensions):**
- `descriptiveness`: Has enough detail for audio generation?
- `realism`: Describes plausible, physically possible audio?
- `coherence`: Internally consistent with no contradictions?
- `completeness`: Covers necessary elements (timing, sounds, atmosphere)?

**Category 4: Overall Training Value (3 dimensions):**
- `learning_signal`: Would training on this teach good behaviors?
- `non_trivial`: Requires actual understanding, not just keyword copying?
- `persona_fit`: Does the persona make sense for this audio type?

Each dimension scored 1-5. Pass criteria: all scores >= 3, average >= 3.5.

**Output**:
- `stage4_quality_judge/judge_{level}.jsonl` - Per-sample scores
- `stage4_quality_judge/summary_{level}.json` - Per-level statistics
- `stage4_quality_judge/summary_all.json` - Combined statistics

### Stage 5: Merge Dialogues

Merges all detail levels into a single file.

**Output**: `stage3_dialogues/dialogues_all.jsonl`

### Stage 6: Generate Statistics

Creates a summary of the pipeline results.

**Output**: `stats.txt`

## Output Structure

```
/work/nvme/bbjs/shared/opuslm_v2_data/sft_data/sft_simulation/
└── {version}/
    ├── stage1_user_requests/
    │   └── user_requests.jsonl
    ├── stage2_reasoning_traces/
    │   ├── reasoning_realistic.jsonl
    │   └── reasoning_imaginary.jsonl
    ├── stage3_dialogues/
    │   ├── dialogues_realistic.jsonl
    │   ├── dialogues_imaginary.jsonl
    │   └── dialogues_all.jsonl
    ├── stage4_quality_judge/
    │   ├── judge_realistic.jsonl
    │   ├── judge_imaginary.jsonl
    │   ├── summary_realistic.json
    │   ├── summary_imaginary.json
    │   └── summary_all.json
    └── stats.txt
```

## Examples

### Full Production Run

```bash
# With multiple vLLM instances for high throughput
./local/sft_data_simulation.sh \
    --version prod_v1 \
    --vllm_url_stage1 "http://gpu1:8000/v1:http://gpu2:8000/v1:http://gpu3:8000/v1" \
    --vllm_url_stage2 "http://gpu4:8000/v1:http://gpu5:8000/v1" \
    --vllm_url_stage4 "http://gpu1:8000/v1:http://gpu2:8000/v1:http://gpu3:8000/v1" \
    --model_stage1 "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8" \
    --model_stage2 "Qwen/Qwen3-8B" \
    --model_stage4 "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8" \
    --num_workers 512
```

### Quick Test

```bash
./local/sft_data_simulation.sh \
    --version test \
    --num_samples 10 \
    --num_workers 8
```

### Resume After Failure

```bash
# Original run failed at Stage 2
./local/sft_data_simulation.sh \
    --version prod_v1 \
    --stage 2 \
    --resume true
```

## Troubleshooting

1. **API Timeout**: Increase `--timeout` for slow responses
2. **Memory Issues**: Reduce `--num_workers` to lower concurrent requests
3. **Resume Not Working**: Ensure `--version` matches the original run
4. **Missing Stages**: Check that previous stage outputs exist before running later stages
