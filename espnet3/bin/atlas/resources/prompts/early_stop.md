Keypoint check for ${trial_id}: ${labels}.
Current progress: ${progress}.

Read current evidence with tools as needed; do not ask for pasted copies:
- Trial directory: ${trial_dir}
- Training log: ${trial_dir}/train.log
- Evaluation log: ${trial_dir}/eval.log
- Live training metrics: ${metrics_csv}
- Trial history and scores: ${study_dir}/trials.csv
- Study objective: ${study_dir}/program.md

Should this trial be early stopped?
Stop if the trajectory is clearly inferior to other trials and unlikely to recover.
Report the metrics behind your decision in key_metrics.
Be conservative — prefer to continue unless the evidence is strong.
