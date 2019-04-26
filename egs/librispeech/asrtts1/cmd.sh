# you can change cmd.sh depending on what type of queue you are using.
# If you have no queueing system and want to run on a local machine, you
# can change all instances 'queue.pl' to run.pl (but be careful and run
# commands one by one: most recipes will exhaust the memory on your
# machine).  queue.pl works with GridEngine (qsub).  slurm.pl works
# with slurm.  Different queues are configured differently, with different
# queue names and different ways of specifying things like memory;
# to account for these differences you can create and edit the file
# conf/queue.conf to match your queue's configuration.  Search for
# conf/queue.conf in http://kaldi-asr.org/doc/queue.html for more information,
# or search for the string 'default_config' in utils/queue.pl or utils/slurm.pl.

export train_cmd="queue.pl --mem 2G -V"
export cuda_cmd="queue.pl -V --gpu 1"
export decode_cmd="queue.pl --mem 4G -V"
CUDAROOT=/usr/local/share/cuda-8.0.61
CUDAROOT=/usr/local/share/cuda-9.0.176

export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT
export CPATH=$CUDA_HOME/include
export PATH=$CUDAROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDAROOT/lib64:$LD_LIBRARY_PATH
export CC="/usr/local/bin/gcc-5.3"
export CC="/usr/local/bin/gcc-6.4"
export CXX="/usr/local/bin/g++-5.3"
export CXX="/usr/local/bin/g++-6.4"


# JHU setup
#export train_cmd="queue.pl --mem 2G"
#export cuda_cmd="queue.pl --mem 2G --gpu 1 --config conf/gpu.conf"
#export decode_cmd="queue.pl --mem 4G"
