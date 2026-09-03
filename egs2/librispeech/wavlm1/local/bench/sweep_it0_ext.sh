#!/usr/bin/env bash
# Extend the iteration-0 ladder upward. The 12M point used only 21.2 GB of
# 79.2, so the ceiling is far above the original 30M top of the ladder.
set -u
cd "$(dirname "$0")/../.."
while pgrep -f "bash local/bench/sweep_it0.sh" >/dev/null 2>&1; do sleep 60; done
sed -e 's/^run_point i0_12M 12000000 1200 false$/run_point i0_36M 36000000 1200 false/' \
    -e 's/^run_point i0_18M 18000000 1200 false$/run_point i0_42M 42000000 1200 false/' \
    -e 's/^run_point i0_24M 24000000 1200 false$/run_point i0_48M 48000000 1200 false/' \
    -e '/^run_point i0_30M 30000000 1200 false$/d' \
    -e 's/iteration-0 batch_bins ladder complete/iteration-0 EXTENDED ladder complete/' \
    local/bench/sweep_it0.sh > local/bench/.sweep_it0_ext_inner.sh
bash local/bench/.sweep_it0_ext_inner.sh
