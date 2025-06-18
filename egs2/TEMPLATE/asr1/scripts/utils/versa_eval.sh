#!/bin/bash
#
# Thread Launcher for VERSA Processing
# -------------------------------------------
# This script splits input audio files and launches Slurm jobs for parallel processing
# using either GPU, CPU, or both resources based on user selection.
#
# Usage: ./versa_eval.sh <pred_wavscp> <gt_wavscp> <gt_text> <score_dir> <split_size> <metric_config> [--cpu-only|--gpu-only]
#   <pred_wavscp>: Path to prediction wav.scp file
#   <gt_wavscp>: Path to ground truth wav.scp file (use "None" if not available)
#   <gt_text>: Path to ground truth text file (use "None" if not available)
#   <score_dir>: Directory to store results
#   <split_size>: Number of chunks to split the data into
#   <metric_config>: Configuration of metrics
#   --cpu-only: Optional flag to run only CPU jobs
#   --gpu-only: Optional flag to run only GPU jobs

# Example: ./versa_eval.sh data/pred.scp data/gt.scp results/experiment1 10 speech.yaml --cpu_only
# Example: ./versa_eval.sh data/pred.scp data/gt.scp results/experiment1 10 speech.yaml --gpu_only

set -e
set -u
set -o pipefail

if ! command -v versa-score >/dev/null 2>&1; then
    echo -e "Repo versa does not install. You can get it at https://github.com/wavlab-speech/versa.git."
    exit 1
fi

# Define color codes for output messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check for minimum required arguments
if [ $# -lt 4 ]; then
    echo -e "${RED}Error: Insufficient arguments${NC}"
    echo -e "${BLUE}Usage: $0 <pred_wavscp> <gt_wavscp> <score_dir> <split_size> [--cpu-only|--gpu-only]${NC}"
    echo -e "  <pred_wavscp>: Path to prediction wav script file"
    echo -e "  <gt_wavscp>: Path to ground truth wav script file (use \"None\" if not available)"
    echo -e "  <score_dir>: Directory to store results"
    echo -e "  <split_size>: Number of chunks to split the data into"
    echo -e "  <gpu_id>: Device id"
    exit 1
fi

# Parse command line arguments
PRED_WAVSCP=$1
GT_WAVSCP=$2
GT_TEXT=$3
SCORE_DIR=$4
SPLIT_SIZE=$5
METRIC_CONFIG=$6

# Default to running both CPU and GPU jobs
use_gpu=false

# Check for optional flags
if [ $# -ge 7 ]; then
    if [ "$7" = "--cpu-only" ]; then
        use_gpu=false
        echo -e "${YELLOW}Running in CPU-only mode${NC}"
    elif [ "$7" = "--gpu-only" ]; then
        use_gpu=true
        echo -e "${YELLOW}Running in GPU-only mode${NC}"
    else
        echo -e "${RED}Error: Unknown option '$7'${NC}"
        echo -e "${BLUE}Valid options are: --cpu-only, --gpu-only${NC}"
        exit 1
    fi
fi

# Validate inputs
if [ ! -f "${PRED_WAVSCP}" ]; then
    echo -e "${RED}Error: Prediction wav script file '${PRED_WAVSCP}' not found${NC}"
    exit 1
fi

if [ "${GT_WAVSCP}" != "None" ] && [ ! -f "${GT_WAVSCP}" ]; then
    echo -e "${RED}Error: Ground truth wav script file '${GT_WAVSCP}' not found${NC}"
    exit 1
fi

if ! [[ "${SPLIT_SIZE}" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: Split size must be a positive integer${NC}"
    exit 1
fi

# Print configuration summary
echo -e "${BLUE}=== Configuration Summary ===${NC}"
echo -e "Prediction WAV script: ${PRED_WAVSCP}"
echo -e "Ground truth WAV script: ${GT_WAVSCP}"
echo -e "Output directory: ${SCORE_DIR}"
echo -e "Split size: ${SPLIT_SIZE}"
if $use_gpu; then
    echo -e "GPU processing: Enabled"
else
    echo -e "CPU processing: Enabled"
fi
echo ""

# Create directory structure
if [ -e "${SCORE_DIR}/pred" ]; then
    rm -r "${SCORE_DIR}/pred"
fi
if [ -e "${SCORE_DIR}/gt" ]; then
    rm -r "${SCORE_DIR}/gt"
fi
echo -e "${GREEN}Creating directory structure...${NC}"
mkdir -p "${SCORE_DIR}"
mkdir -p "${SCORE_DIR}/pred"
mkdir -p "${SCORE_DIR}/gt"
mkdir -p "${SCORE_DIR}/result"
mkdir -p "${SCORE_DIR}/logs"

# Split prediction files
total_lines=$(wc -l < "${PRED_WAVSCP}")

echo -e "${GREEN}Splitting ${total_lines} lines into ${SPLIT_SIZE} pieces${NC}"
source_wavscp=$(basename "${PRED_WAVSCP}")
_split_files=""
for n in $(seq ${SPLIT_SIZE}); do
    _split_files+="${SCORE_DIR}/pred/${source_wavscp}.${n} "
done
utils/split_scp.pl ${PRED_WAVSCP} ${_split_files}
pred_list=("${SCORE_DIR}/pred/${source_wavscp}."*)

# Split ground truth files if provided
if [ "${GT_WAVSCP}" = "None" ]; then
    echo -e "${YELLOW}No ground truth audio provided, evaluation will be reference-free${NC}"
    gt_list=()
else
    target_wavscp=$(basename "${GT_WAVSCP}")
    _split_files=""
    for n in $(seq ${SPLIT_SIZE}); do
        _split_files+="${SCORE_DIR}/gt/${target_wavscp}.${n} "
    done
    utils/split_scp.pl ${GT_WAVSCP} ${_split_files}
    gt_list=("${SCORE_DIR}/gt/${target_wavscp}."*)

    if [ ${#pred_list[@]} -ne ${#gt_list[@]} ]; then
        echo -e "${RED}Error: In wav, the number of split ground truth (${#gt_list[@]}) and predictions (${#pred_list[@]}) does not match.${NC}"
        exit 1
    fi
fi

# Split ground truth files if provided
if [ "${GT_TEXT}" = "None" ]; then
    echo -e "${YELLOW}No ground truth audio provided, evaluation will be reference-free${NC}"
    gt_text_list=()
else
    target_text=$(basename "${GT_TEXT}")
    _split_files=""
    for n in $(seq ${SPLIT_SIZE}); do
        _split_files+="${SCORE_DIR}/gt/${target_text}.${n} "
    done
    utils/split_scp.pl ${GT_TEXT} ${_split_files}
    gt_list=("${SCORE_DIR}/gt/${target_text}."*)

    if [ ${#pred_list[@]} -ne ${#gt_text_list[@]} ]; then
        echo -e "${RED}Error: In text, the number of split ground truth (${#gt_text_list[@]}) and predictions (${#pred_list[@]}) does not match.${NC}"
        exit 1
    fi
fi

pids=()
for ((i=0; i<${#pred_list[@]}; i++))  ; do
(
    sub_pred_wavscp=${pred_list[${i}]}
    job_prefix="${sub_pred_wavscp##*.}"

    opts=""
    if [ "${GT_WAVSCP}" != "None" ]; then
        opts+="--gt ${gt_list[${i}]} "
    fi
    if [ "${GT_TEXT}" != "None" ]; then
        opts+="--text ${gt_text_list[${i}]} "
    fi

    echo -e "${BLUE}Processing chunk $((i+1))/${#pred_list[@]}: ${sub_pred_wavscp}${NC}"

    logfile="${SCORE_DIR}/logs/eval.$job_prefix.log"
    true > ${logfile}
    versa-score --score_config ${METRIC_CONFIG} \
        --pred "${sub_pred_wavscp}" \
        --output_file "${SCORE_DIR}/result/result.$job_prefix.txt" \
        --io soundfile \
        --use_gpu ${use_gpu} ${opts} >> "$logfile" 2>&1 || {
            echo "[JOB ${job_prefix}] Failed.";
            tail -n 50 "$logfile";
            exit 1;
        }
    echo "[JOB ${job_prefix}] Finished." >> "$logfile"
) &
    pids+=($!)
done

i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
[ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;

echo -e "${GREEN}Successfully finish evaluation on ${METRIC_CONFIG}! ${NC}"
