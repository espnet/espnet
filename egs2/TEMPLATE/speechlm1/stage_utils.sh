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

latest_dcp() {
    # The highest step_N under a checkpoints directory that finished writing.
    # A DCP without .metadata is an interrupted checkpoint, and reading one
    # fails deep inside the loader rather than here.
    local directory="$1" candidate best=
    for candidate in "${directory}"/step_*; do
        [[ -f ${candidate}/.metadata ]] || continue
        if [[ -z ${best} || ${candidate##*step_} -gt ${best##*step_} ]]; then
            best=${candidate}
        fi
    done
    [[ -n ${best} ]] || return 1
    echo "${best}"
}

resume_argument() {
    # How a training stage starts: from its own checkpoints if it has any,
    # which is what continuing an interrupted stage needs, and otherwise
    # from the stage before it. Prints the argument, or nothing.
    local output="$1" previous="$2"
    if latest_dcp "${output}/checkpoints" >/dev/null 2>&1; then
        log "${output} already has checkpoints; continuing it"
        return 0
    fi
    [[ -n ${previous} ]] || return 0
    local started_from
    started_from=$(latest_dcp "${previous}/checkpoints") \
        || die "no complete checkpoint under ${previous}/checkpoints to start from"
    log "starting from ${started_from}"
    echo "--resume-path ${started_from}"
}
