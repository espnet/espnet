#!/usr/bin/env bats



@test "eval_turn_taking_metric" {
    cd egs2/swbd/asr1
    run pytest pyscripts/utils/test_compute_turn_take_metrics.py
    echo "$output"  # Print pytest output for debugging

    # Ensure pytest ran successfully (exit code 0 means all tests passed)
    [ "$status" -eq 0 ]
}
