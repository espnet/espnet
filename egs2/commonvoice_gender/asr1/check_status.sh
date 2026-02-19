#!/bin/bash
# Quick status check for female ASR job. Usage: ./check_status.sh [job_id]
# With no args, uses latest slurm_female_*.log
JOB_ID="${1:-}"
if [ -n "$JOB_ID" ]; then
  LOG="slurm_female_${JOB_ID}.log"
else
  LOG=$(ls -t slurm_female_*.log 2>/dev/null | head -1)
fi
echo "=== Queue ==="
squeue -u $USER
echo ""
echo "=== Job history (last job) ==="
sacct -u $USER --format=JobID,JobName,State,ExitCode,Start,End,Elapsed -n | head -5
echo ""
if [ -n "$LOG" ] && [ -f "$LOG" ]; then
  echo "=== Last 25 lines of $LOG ==="
  tail -25 "$LOG"
else
  echo "=== No log file found yet (job may not have started) ==="
fi
