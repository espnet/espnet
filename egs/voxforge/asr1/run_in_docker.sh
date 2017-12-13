#!/bin/bash

gpu=0
name="chainer300_cudnn7.0_nossh"
image="chainer/3.0.0nossh:9.0-cudnn7-16.04"

while test $# -gt 0
do
    case "$1" in
        --gpu) gpu=$2
            ;;
        --*) echo "bad option $1"
            exit 1;;
        *) echo "argument $1"
            exit 1;;
    esac
    shift
    shift
done

docker_image=$( docker images -q $image ) 

if ! [[ -n $docker_image  ]]; then
  voxforge=$PWD
  cd ../../../tools
  echo "Building docker image..."
  (docker build -f "$name".devel -t $image .) || exit 1
  cd $voxforge
fi

vol1="$PWD/../../../src:/spnet/src"
vol2="$PWD/../../../egs:/spnet/egs"

cmd1="cd /spnet/egs/voxforge/asr1"
cmd2="./run.sh --docker true"
cmd3="chmod -R 777 /spnet/egs/voxforge/asr1" #Required to access once the training if finished

if [ ${gpu} -le -1 ]; then
  cmd="docker run -i --rm --name spnet_nogpu -v $vol1 -v $vol2 $image /bin/bash -c '$cmd1; $cmd2; $cmd3'"
else
  cmd="NV_GPU=$gpu nvidia-docker run -i --rm --name spnet$gpu -v $vol1 -v $vol2 $image /bin/bash -c '$cmd1; $cmd2; $cmd3'" 
	# --rm erase the container when the training is finished.
fi

echo "Executing application in Docker"
eval $cmd

echo "`basename $0` done."
