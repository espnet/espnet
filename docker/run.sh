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

if [ -z "${docker_egs}" ]; then
  echo "Select an example to work with from the egs folder."
  exit 1
fi

from_image="ubuntu:16.04"
image_label="espnet:ubuntu16.04"
if [ ! "${docker_gpu}" == "-1" ]; then
  cuda_ver=$( nvcc -V | grep release )
  cuda_ver=${cuda_ver#*"release "}
  cuda_ver=${cuda_ver%,*}
  if [ ! -z "${cuda_ver}" ]; then
    cudnn_ver=$( cat /usr/local/cuda/include/cudnn.h | grep "#define CUDNN_MAJOR" )
    cudnn_ver=${cudnn_ver#*MAJOR}
    cudnn_ver=${cudnn_ver// /}
    if [ ! -z "${cudnn_ver}" ]; then
      from_image="nvidia/cuda:${cuda_ver}-cudnn${cudnn_ver}-devel-ubuntu16.04"
      image_label="espnet:cuda${cuda_ver}-cudnn${cudnn_ver}-ubuntu16.04"
    else
      echo "CUDNN was not found in default folder."
      from_image="nvidia/cuda:${cuda_ver}-devel-ubuntu16.04"
      image_label="espnet:cuda${cuda_ver}-ubuntu16.04"
    fi
  else
    echo "CUDA version was not found, selecting CPU image. For GPU image, install NVIDIA-DOCKER, CUDA and NVCC."
  fi
fi
echo "Using image ${from_image}."
build_args="--build-arg FROM_IMAGE=${from_image}"
if [ ! -z "${HTTP_PROXY}" ]; then
  echo "Building with proxy ${HTTP_PROXY}"
  build_args="${build_args} --build-arg WITH_PROXY=${HTTP_PROXY}"
fi 

docker_image=$( docker images -q ${image_label} ) 

cd ..
if ! [[ -n ${docker_image}  ]]; then
  echo "Building docker image..."
  (docker build ${build_args} -f docker/espnet.devel -t ${image_label} .) || exit 1
fi

vols="-v $PWD/egs:/espnet/egs -v $PWD/src:/espnet/src -v $PWD/test:/espnet/test"
if [ ! -z "${docker_folders}" ]; then
  docker_folders=$(echo ${docker_folders} | tr "," "\n")
  for i in ${docker_folders[@]}
  do
    vols=${vols}" -v $i:$i";
  done
fi

cmd1="cd /espnet/egs/${docker_egs}"
cmd2="./run.sh $@"
#Required to access to the folder once the training if finished
cmd3="chmod -R 777 /espnet/egs/${docker_egs}"

cmd="${cmd1}; ${cmd2}; ${cmd3}"
if [ "${docker_gpu}" == "-1" ]; then
  cmd="docker run -i --rm --name espnet_cpu ${vols} ${image_label} /bin/bash -c '${cmd}'"
else
  # --rm erase the container when the training is finished.
  cmd="NV_GPU='${docker_gpu}' nvidia-docker run -i --rm --name espnet_gpu${docker_gpu} ${vols} ${image_label} /bin/bash -c '${cmd}'"
fi

echo "Executing application in Docker"
echo ${cmd}
eval ${cmd}

echo "`basename $0` done."
