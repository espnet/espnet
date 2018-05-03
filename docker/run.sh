#!/bin/bash

docker_gpu=0
docker_egs=
docker_folders=

while test $# -gt 0
do
    case "$1" in
        -h) echo "Usage: `basename $0` [-h] docker_gpu docker_egs docker_folders options"
            exit 0;;
        --help) echo "Usage: `basename $0` [-h] ] docker_gpu docker_egs docker_folders options"
              exit 0;;
        --docker*) ext=${1#--}
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
        --*) break
              ;;
    esac
    shift
    shift
done

if [ -z "$docker_egs" ]; then
  echo "Select an example to work with from the folder egs"
  exit 1
fi

image="espnet/1.0:9.0-cudnn7-16.04"
docker_image=$( docker images -q $image ) 

if ! [[ -n $docker_image  ]]; then
  echo "Building docker image..."
  # If you have already run a egs it will take time to load all the data from the egs.
  # To avoid it, it is better to copy the folders, rather that run from the parent folder
  cp -r ../src ../test ../tools ./
  (docker build -f espnet.devel -t $image .) || exit 1
  rm -r ./src ./test ./tools
fi

vols="-v $PWD/../egs:/espnet/egs"
if [ ! -z "$docker_folders" ]; then
  docker_folders=$(echo $docker_folders | tr "," "\n")
  for i in ${docker_folders[@]}
  do
    vols=$vols" -v $i:$i";
  done
fi

cmd1="cd /espnet/egs/$docker_egs/asr1"
cmd2="./run.sh $@"
#Required to access to the folder once the training if finished
cmd3="chmod -R 777 /espnet/egs/$docker_egs/asr1"

if [ ${gpu} -le -1 ]; then
  cmd="docker run -i --rm --name espnet_nogpu $vols $image /bin/bash -c '$cmd1; $cmd2; $cmd3'"
else
  # --rm erase the container when the training is finished.
  cmd="NV_GPU='$docker_gpu' nvidia-docker run -i --rm --name espnet_gpu$$docker_gpu $vols $image /bin/bash -c '$cmd1; $cmd2; $cmd3'"
fi

echo "Executing application in Docker"
echo $cmd
eval $cmd

echo "`basename $0` done."
