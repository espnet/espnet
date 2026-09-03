# ====== About run.pl, queue.pl, slurm.pl, and ssh.pl ======
# Usage: <cmd>.pl [options] JOB=1:<nj> <log> <command...>
# e.g.
#   run.pl --mem 4G JOB=1:10 echo.JOB.log echo JOB
#
# Options:
#   --time <time>: Limit the maximum time to execute.
#   --mem <mem>: Limit the maximum memory usage.
#   -–max-jobs-run <njob>: Limit the number parallel jobs. This is ignored for non-array jobs.
#   --num-threads <ngpu>: Specify the number of CPU core.
#   --gpu <ngpu>: Specify the number of GPU devices.
#   --config: Change the configuration file from default.
#
# "JOB=1:10" is used for "array jobs" and it can control the number of parallel jobs.
# The left string of "=", i.e. "JOB", is replaced by <N>(Nth job) in the command and the log file name,
# e.g. "echo JOB" is changed to "echo 3" for the 3rd job and "echo 8" for 8th job respectively.
# Note that the number must start with a positive number, so you can't use "JOB=0:10" for example.
#
# run.pl, queue.pl, slurm.pl, and ssh.pl have unified interface, not depending on its backend.
# These options are mapping to specific options for each backend and
# it is configured by "conf/queue.conf" and "conf/slurm.conf" by default.
# If jobs failed, your configuration might be wrong for your environment.
#
#
# The official documentation for run.pl, queue.pl, slurm.pl, and ssh.pl:
#   "Parallelization in Kaldi": http://kaldi-asr.org/doc/queue.html
# =========================================================~


# Select the backend used by run.sh from "local", "stdout", "sge", "slurm", or "ssh"
cmd_backend='slurm'

# Local machine, without any Job scheduling system
if [ "${cmd_backend}" = local ]; then

    # The other usage
    export train_cmd="run.pl"
    # Used for "*_train.py": "--gpu" is appended optionally by run.sh
    export cuda_cmd="run.pl"
    # Used for "*_recog.py"
    export decode_cmd="run.pl"

# Local machine logging to stdout and log file, without any Job scheduling system
elif [ "${cmd_backend}" = stdout ]; then

    # The other usage
    export train_cmd="stdout.pl"
    # Used for "*_train.py": "--gpu" is appended optionally by run.sh
    export cuda_cmd="stdout.pl"
    # Used for "*_recog.py"
    export decode_cmd="stdout.pl"


# "qsub" (Sun Grid Engine, or derivation of it)
elif [ "${cmd_backend}" = sge ]; then
    # The default setting is written in conf/queue.conf.
    # You must change "-q g.q" for the "queue" for your environment.
    # To know the "queue" names, type "qhost -q"
    # Note that to use "--gpu *", you have to setup "complex_value" for the system scheduler.

    export train_cmd="queue.pl"
    export cuda_cmd="queue.pl"
    export decode_cmd="queue.pl"


# "qsub" (Torque/PBS.)
elif [ "${cmd_backend}" = pbs ]; then
    # The default setting is written in conf/pbs.conf.

    export train_cmd="pbs.pl"
    export cuda_cmd="pbs.pl"
    export decode_cmd="pbs.pl"


# "sbatch" (Slurm)
elif [ "${cmd_backend}" = slurm ]; then
    # The default setting is written in conf/slurm.conf.
    # You must change "-p cpu" and "-p gpu" for the "partition" for your environment.
    # To know the "partion" names, type "sinfo".
    # You can use "--gpu * " by default for slurm and it is interpreted as "--gres gpu:*"
    # The devices are allocated exclusively using "${CUDA_VISIBLE_DEVICES}".

    # NOTE: honour a pre-set train_cmd/cuda_cmd from the environment. wavlm.sh
    # sources this file AFTER utils/parse_options.sh, so an unconditional export
    # here silently clobbers any --*_cmd passed on the command line, and
    # ${kmeans_opts} cannot carry a multi-word value (it is word-split at
    # wavlm.sh:417). This is the only override point that works, e.g.
    #   train_cmd="slurm.pl --config conf/slurm_cpu_on_gpu.conf" ./run_large.sh ...
    # NOTE (2026-09-01): that routing NO LONGER WORKS. The gpu partition now
    # rejects CPU-only jobs outright:
    #   "the gpu partition allocates whole GPUs (24 CPUs + 115200 MB each):
    #    request GPUs (e.g. --gpus=1) or submit CPU-only jobs with -p cpu"
    # It was accepted earlier the same night, so the policy changed mid-run --
    # plausibly because a 128-job CPU-only array sat on those nodes for 4 h.
    # CPU work goes to the 64-core `cpu` partition; size nj accordingly.
    # to put CPU-only jobs on the 208-core gpu nodes instead of the 64-core
    # `cpu` partition. Defaults are unchanged when nothing is pre-set.
    export train_cmd="${train_cmd:-slurm.pl}"
    export cuda_cmd="${cuda_cmd:-slurm.pl}"
    export decode_cmd="slurm.pl"

elif [ "${cmd_backend}" = ssh ]; then
    # You have to create ".queue/machines" to specify the host to execute jobs.
    # e.g. .queue/machines
    #   host1
    #   host2
    #   host3
    # Assuming you can login them without any password, i.e. You have to set ssh keys.

    export train_cmd="ssh.pl"
    export cuda_cmd="ssh.pl"
    export decode_cmd="ssh.pl"

# This is an example of specifying several unique options in the JHU CLSP cluster setup.
# Users can modify/add their own command options according to their cluster environments.
elif [ "${cmd_backend}" = jhu ]; then

    export train_cmd="queue.pl --mem 2G"
    export cuda_cmd="queue-freegpu.pl --mem 2G --gpu 1 --config conf/queue.conf"
    export decode_cmd="queue.pl --mem 4G"

else
    echo "$0: Error: Unknown cmd_backend=${cmd_backend}" 1>&2
    return 1
fi
