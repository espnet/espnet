# ====== About run.pl, queue.pl, slurm.pl, and ssh.pl ======
# Usage: <cmd>.pl [options] JOB=1:<nj> <log> <command...>
# e.g.
#   run.pl --mem 4G JOB=1:10 echo.JOB.log echo JOB
#
# Options:
#   --time <time>: Limit the maximum time to execute.
#   --mem <mem>: Limit the maximum memory usage.
#   -窶杜ax-jobs-run <njob>: Limit the number parallel jobs. This is ignored for non-array jobs.
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
# The official documentaion for run.pl, queue.pl, slurm.pl, and ssh.pl:
#   "Parallelization in Kaldi": http://kaldi-asr.org/doc/queue.html
