#!/bin/bash

gpu=0
backend=
egs=
stage=
corpus_dir=
#egs_opts bypasses the arguments for each specific egs
egs_opts=

while test $# -gt 0
do
    case "$1" in
        -h) echo "Usage: `basename $0` [-h] gpu model rotation type"
            exit 0;;
        --help) echo "Usage: `basename $0` [-h] gpu model rotation type"
              exit 0;;
        --egs_opts) egs_opts="$2";;
        --*) ext=${1#--}
              frombreak=true
              for i in _ {a..z} {A..Z}; do
                for var in `eval echo "\\${!$i@}"`; do
                  if [ "$var" == "$ext" ]; then
                    eval $ext=$2
                    frombreak=false
                    break 2
                  fi 
                done 
              done
              if $frombreak ; then
                echo "bad option $1" 
                exit 1
              fi
              ;;
        *) echo "argument $1 does not exit"
            exit 1;;
    esac
    shift
    shift
done

if [ -z "$backend" ]; then
  echo "Need to specify a backend"
  exit 1
fi

if [ -z "$egs" ]; then
  echo "Select a egs to work with"
  exit 1
fi

image="espnet/1.0:9.0-cudnn7-16.04"
docker_image=$( docker images -q $image ) 

if ! [[ -n $docker_image  ]]; then
  echo "Building docker image..."
  # If you have already run a egs it will take time to load all the data from the egs
  # to avoid it, is better to copy the folders, rather that run from the parent folder
  cp -r ../src ../test ../tools ./
  (docker build -f espnet.devel -t $image .) || exit 1
  rm -r ./src ./test ./tools
fi

vols="-v $PWD/../egs:/espnet/egs"
if [ ! -z "$corpus_dir" ]; then
  vols=$vols" -v $corpus_dir:/$egs"
fi 

cmd1="cd /espnet/egs/$egs/asr1"
cmd2="./run.sh --backend $backend"
if [ ! -z "$stage" ]; then
  cmd2=$cmd2" --stage $stage"
fi

if [ ! -z "$egs_opts" ]; then
  cmd2=$cmd2" $egs_opts"
fi

#Required to access to the folder once the training if finished
cmd3="chmod -R 777 /espnet/egs/$egs/asr1"

if [ ${gpu} -le -1 ]; then
  cmd="docker run -i --rm --name espnet_nogpu $vols $image /bin/bash -c '$cmd1; $cmd2; $cmd3'"
else
  #Current implementation only supportes single GPU, TODO: multiple GPUs
  cmd2=$cmd2" --gpu 0" 
  # --rm erase the container when the training is finished.
  cmd="NV_GPU=$gpu nvidia-docker run -i --rm --name espnet_gpu$gpu $vols $image /bin/bash -c '$cmd1; $cmd2; $cmd3'"
fi

echo "Executing application in Docker"
echo $cmd
eval $cmd


echo "`basename $0` done."
