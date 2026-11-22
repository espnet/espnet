#!/usr/bin/env bash

# Helpers for the SpeechLM recipes' staged run.sh.
# Source it; do not run it.

log() {
    # stderr, so that a helper can report progress while its stdout is
    # being captured by the caller
    echo "$(date '+%Y-%m-%dT%H:%M:%S') $*" >&2
}

die() {
    echo "$*" >&2
    exit 1
}

validate_stages() {
    local stage=$1 stop_stage=$2 last_stage=$3 node_rank=$4
    [[ ${stage} =~ ^[1-9][0-9]*$ && ${stop_stage} =~ ^[1-9][0-9]*$ ]] \
        || die "--stage and --stop-stage must be positive integers"
    (( stage <= stop_stage && stop_stage <= last_stage )) \
        || die "select stages in order between 1 and ${last_stage}"
    [[ ${node_rank} =~ ^(0|[1-9][0-9]*)$ ]] \
        || die "--node-rank must be a nonnegative integer"
}

latest_dcp() {
    # The highest step_N under a checkpoints directory that finished writing.
    # A DCP without .metadata is an interrupted checkpoint, and reading one
    # fails deep inside the loader rather than here.
    local directory="$1" candidate step best='' best_step=-1
    for candidate in "${directory}"/step_*; do
        [[ -f ${candidate}/.metadata ]] || continue
        step=${candidate##*/step_}
        [[ ${step} =~ ^[0-9]+$ ]] || continue
        step=$((10#${step}))
        if (( step > best_step )); then
            best=${candidate}
            best_step=${step}
        fi
    done
    [[ -n ${best} ]] || return 1
    echo "${best}"
}

resume_checkpoint() {
    # How a training stage starts: from its own checkpoints if it has any,
    # which is what continuing an interrupted stage needs, and otherwise
    # from the stage before it. Prints only the path, so callers can quote it.
    # Explicit weight initialization overrides automatic resume, as in train.sh.
    local output="$1" previous="$2" explicit="${3:-}"
    if [[ -n ${explicit} ]]; then
        log "starting from ${explicit}"
        printf '%s\n' "${explicit}"
        return 0
    fi
    if latest_dcp "${output}/checkpoints" >/dev/null 2>&1; then
        log "${output} already has checkpoints; continuing it"
        return 0
    fi
    [[ -n ${previous} ]] || return 0
    local started_from
    started_from=$(latest_dcp "${previous}/checkpoints") \
        || die "no complete checkpoint under ${previous}/checkpoints to start from"
    log "starting from ${started_from}"
    printf '%s\n' "${started_from}"
}
