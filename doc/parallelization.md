# Using Job scheduling system

Our recipes support some Job scheduling systems, SGE, PBS/Torque, 
and Slurm, according to [Parallelization in Kaldi](https://kaldi-asr.org/doc/queue.html). 
By default, the feature extraction and decoding run at the local machine with multiprocessing, 
but if there are any Job scheduling systems in your environment, 
you can submit more number of Jobs with multiple machines. 
Please ask the administrator to install it.

## Select Job scheduler

`cmd.sh` is a configuration file and it's used by `run.sh` to set some shell variables. These shell variables should be set as one of following perl scripts:

|cmd   |Backend                 | configuration file|
|--------| :--------------------------------------:| :---------------: |
|run.pl | Local machine (default)         |-         |
|queue.pl|Sun grid engine, or grid endine like tool|conf/queue.conf  |
|slurm.pl|Slurm                  |conf/slurm.conf  |
|pbs.pl |PBS/Torque                |conf/pbs.conf   |
|ssh.pl |SSH                   |.queue/machines  |



## Usage of run.pl

`run.pl`, `queue.pl`, `slurm.pl`, `pbs.pl` and `ssh.pl` have a unified interface, 
therefore we can assign any one of them to `${cmd}` in the shell script:

```bash
nj=4
${cmd} JOB=1:${nj} JOB.log echo JOB
```

`JOB=1:${nj}` indicates the parallelization, which is known as "array-job", with `${nj}` number of jobs. 
`JOB.log` is a destination of the stdout and stderr from jobs. 
The string of `JOB` will be changed to the job number 
if it's included in the log file name or command line arguments. 
i.e. The following commands are almost equivalent to the above:

```bash
echo 1 &> 1.log &
echo 2 &> 2.log &
echo 3 &> 3.log &
echo 4 &> 4.log &
wait
```

## Queue/Partition setting
You also need to modify the configuration file for a specific job scheduler to change command-line options to submit jobs e.g. queue setting, resource request, etc.

The following text is an example of `conf/queue.conf`.

```
# Default configuration
command qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64*
option mem=* -l mem_free=$0,ram_free=$0
option mem=0     # Do not add anything to qsub_opts
option num_threads=* -pe smp $0
option num_threads=1 # Do not add anything to qsub_opts
option max_jobs_run=* -tc $0
default gpu=0
option gpu=0
option gpu=* -l gpu=$0 -q g.q
```

Take a look at the following:

```
option gpu=* -l gpu=$0 -q g.q
```

This line means that the optional argument specified by the second column, `gpu=*`, 
will be changed the options after it, `gpu=$0 -q g.q`:

```bash
queue.pl --gpu 2
```

will be converted to

```bash
qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -l gpu=2 -q g.q
```

Thanks to this system, our shell script can switch any job scheduling system with unified options.
