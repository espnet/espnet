#!/usr/bin/env bash
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Script for preparing autoencoder evaluation dataset from inference results

set -euo pipefail

# Default values
task_type="auto"
original_name=""
log_level="INFO"

# Usage function
usage() {
    echo "Usage: $0 --decode_dir <path> [--task_type <type>] [--original_name <name>]"
    echo ""
    echo "Arguments:"
    echo "  --decode_dir     Path to decode output directory (required)"
    echo "  --task_type      Task type: audio_to_text, text_to_audio, or auto (default: auto)"
    echo "  --original_name  Original dataset name (extracted from decode_dir if not provided)"
    echo "  --log_level      Logging level: ERROR, WARNING, INFO, DEBUG (default: INFO)"
    echo ""
    echo "Output:"
    echo "  Prints test specifier: {inverse_task}:{name}_autoencoder:{dataset_json_path}"
    echo ""
    echo "Example:"
    echo "  $0 --decode_dir exp/inference/asr_librispeech"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --decode_dir)
            decode_dir="$2"
            shift 2
            ;;
        --task_type)
            task_type="$2"
            shift 2
            ;;
        --original_name)
            original_name="$2"
            shift 2
            ;;
        --log_level)
            log_level="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check required arguments
if [[ -z "${decode_dir:-}" ]]; then
    echo "Error: --decode_dir is required"
    usage
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ESPNET_ROOT="${SCRIPT_DIR}/../../../.."

# Paths to scripts
PREPARE_INPUT_SCRIPT="${SCRIPT_DIR}/prepare_autoencoder_input.py"
PREPARE_DATASET_SCRIPT="${ESPNET_ROOT}/espnet2/speechlm/bin/prepare_dataset_json.py"

# Check if scripts exist
if [[ ! -f "${PREPARE_INPUT_SCRIPT}" ]]; then
    echo "Error: prepare_autoencoder_input.py not found at ${PREPARE_INPUT_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "${PREPARE_DATASET_SCRIPT}" ]]; then
    echo "Error: prepare_dataset_json.py not found at ${PREPARE_DATASET_SCRIPT}" >&2
    exit 1
fi

echo "==============================================" >&2
echo "Autoencoder Evaluation Dataset Preparation" >&2
echo "==============================================" >&2
echo "Decode directory: ${decode_dir}" >&2
echo "Task type: ${task_type}" >&2
echo "" >&2

# Build optional arguments
optional_args=""
if [[ -n "${original_name}" ]]; then
    optional_args="--original_name ${original_name}"
fi

# Step 1: Generate dialogue JSONL from results and get specifier
echo "[Step 1] Generating dialogue JSONL from results.json files..." >&2
specifier=$(python3 "${PREPARE_INPUT_SCRIPT}" \
    --decode_dir "${decode_dir}" \
    --task_type "${task_type}" \
    ${optional_args} \
    --log_level "${log_level}")

# The input_jsonl path is fixed
output_dir="${decode_dir}/autoencoder_eval"
input_jsonl="${output_dir}/input.jsonl"
output_json="${output_dir}/dataset.json"

echo "Generated dialogue JSONL: ${input_jsonl}" >&2

# Step 2: Create dataset JSON using prepare_dataset_json.py
echo "" >&2
echo "[Step 2] Creating dataset JSON..." >&2
python3 "${PREPARE_DATASET_SCRIPT}" \
    --triplets "dialogue,${input_jsonl},dialogue" \
    --output_json "${output_json}" \
    --log_level "${log_level}"

echo "" >&2
echo "[Step 3] Adding entry to autoencoder registry..." >&2

# Parse specifier to get inverse_task and dataset_name
# Specifier format: ${inverse_task}:${name}_autoencoder:${path}
inverse_task=$(echo "${specifier}" | cut -d':' -f1)
dataset_name=$(echo "${specifier}" | cut -d':' -f2 | sed 's/_autoencoder$//')

# Create registry entry name: ${dataset_name}_${inverse_task}
registry_name="${dataset_name}_${inverse_task}"
# Put registry file in parent directory of decode_dir
registry_dir="$(dirname "${decode_dir}")"
registry_file="${registry_dir}/autoencoder_registry.yaml"

# Create file if it doesn't exist
if [[ ! -f "${registry_file}" ]]; then
    touch "${registry_file}"
    echo "Created new registry file: ${registry_file}" >&2
fi

# Add entry to YAML file (append) in the format:
# dataset_name:
#     path: /path/to/data.json
{
    echo ""
    echo "${registry_name}:"
    echo "    path: ${output_json}"
} >> "${registry_file}"
echo "Added entry: ${registry_name}" >&2

echo "" >&2
echo "==============================================" >&2
echo "Done!" >&2
echo "==============================================" >&2
echo "Output files:" >&2
echo "  - Dialogue JSONL: ${input_jsonl}" >&2
echo "  - Dataset JSON:   ${output_json}" >&2
echo "  - Registry YAML:  ${registry_file}" >&2
echo "" >&2
echo "Test specifier:" >&2
echo "  ${specifier}" >&2

# Output specifier to stdout for capture
echo "${specifier}"
