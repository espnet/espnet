# Mutual exclusion between the batch chain and the interactive chain.
#
# Both write the same exp_dir. Two concurrent trainers would interleave writes
# to last.ckpt and the periodic checkpoints, and the loser's steps would be
# silently discarded on the next resume -- so exactly one may run at a time.
#
# The lock is a DIRECTORY, because mkdir is atomic on a shared filesystem
# whereas "test -f then touch" is not: two jobs starting in the same second
# would both see it missing and both proceed.
#
# Staleness is decided by asking SLURM, not by a timeout: the lock records the
# job id that took it, and a lock whose job is no longer in the queue is dead
# and gets cleared. squeue only shows our own jobs on this cluster, which is
# all that is needed since only our own jobs ever take this lock.
#
# Usage:  . train_lock.sh ; acquire_lock <dir> <holder> <max_wait_s> ; release_lock <dir> <holder>

acquire_lock() {
    local lockdir="$1" holder="$2" max_wait="${3:-0}" waited=0
    while : ; do
        if mkdir "$lockdir" 2>/dev/null; then
            echo "$holder" > "$lockdir/holder"
            return 0
        fi
        local owner
        owner=$(cat "$lockdir/holder" 2>/dev/null || echo "")
        if [ -z "$owner" ]; then
            # half-created lock; treat as stale after a moment
            sleep 2; rmdir "$lockdir" 2>/dev/null || rm -rf "$lockdir" 2>/dev/null
            continue
        fi
        if ! squeue -j "$owner" -h -o "%i" 2>/dev/null | grep -q .; then
            echo "  lock held by job $owner which is no longer queued -- clearing" >&2
            rm -rf "$lockdir"
            continue
        fi
        if [ "$waited" -ge "$max_wait" ]; then
            echo "  lock held by live job $owner; gave up after ${waited}s" >&2
            return 1
        fi
        sleep 30; waited=$((waited + 30))
    done
}

release_lock() {
    local lockdir="$1" holder="$2"
    local owner
    owner=$(cat "$lockdir/holder" 2>/dev/null || echo "")
    # only the holder may release, so a job exiting late cannot free someone
    # else's lock
    if [ "$owner" = "$holder" ]; then
        rm -rf "$lockdir"
    fi
}
