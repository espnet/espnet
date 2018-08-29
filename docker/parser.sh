#!/bin/bash

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

if [ -z "${docker_egs}" ]; then
  echo "Select an example to work with from the egs folder."
  exit 1
fi

from_image="ubuntu:16.04"
image_label="espnet/espnet:cpu"
if [ ! "${docker_gpu}" == "-1" ]; then
  if [ -z "${docker_cuda}" ]; then
    # If the docker_cuda is not set, the program will automatically 
    # search the installed version with default configurations (apt)
    docker_cuda=$( nvcc -V | grep release )
    docker_cuda=${docker_cuda#*"release "}
    docker_cuda=${docker_cuda%,*}
    if [ ! -z "${docker_cuda}" ]; then
      docker_cudnn=$( cat /usr/local/cuda/include/cudnn.h | grep "#define CUDNN_MAJOR" )
      docker_cudnn=${docker_cudnn#*MAJOR}
      docker_cudnn=${docker_cudnn// /}
      if [ ! -z "${docker_cudnn}" ]; then
        from_image="nvidia/cuda:${docker_cuda}-cudnn${docker_cudnn}-devel-ubuntu16.04"
        image_label="espnet/espnet:cuda${docker_cuda}-cudnn${docker_cudnn}-ubuntu16.04"
      else
        echo "CUDNN was not found in default folder."
        from_image="nvidia/cuda:${docker_cuda}-devel-ubuntu16.04"
        image_label="espnet/espnet:cuda${docker_cuda}-ubuntu16.04"
      fi
    else
      echo "CUDA was not found, selecting CPU image. For GPU image, install NVIDIA-DOCKER, CUDA and NVCC."
    fi
  else
    from_image="nvidia/cuda:${docker_cuda}-cudnn${docker_cudnn}-devel-ubuntu16.04"
    image_label="espnet/espnet:cuda${docker_cuda}-cudnn${docker_cudnn}-ubuntu16.04"
  fi
fi
if [ ${docker_user} -gt 0 ]; then
  image_label="${image_label}-user"
fi
docker_file="docker/Dockerfile"

echo "Using image ${from_image}."
docker_image=$( docker images -q ${image_label} ) 

this_time="$(date '+%Y%m%dT%H%M')"
if [ "${docker_gpu}" == "-1" ]; then
  cmd0="docker"
  container_name="espnet_cpu_${this_time}"
else
  # --rm erase the container when the training is finished.
  cmd0="NV_GPU='${docker_gpu}' nvidia-docker"
  container_name="espnet_gpu${docker_gpu//,/_}_${this_time}"
fi

true; # so this script returns exit code 0.