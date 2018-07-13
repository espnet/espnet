#!/bin/bash

docker_gpu=0
docker_egs=
docker_folders=
docker_cuda=9.0
docker_cudnn=7
docker_user=0

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
        image_label="espnet:cuda${docker_cuda}-cudnn${docker_cudnn}-ubuntu16.04"
      else
        echo "CUDNN was not found in default folder."
        from_image="nvidia/cuda:${docker_cuda}-devel-ubuntu16.04"
        image_label="espnet:cuda${docker_cuda}-ubuntu16.04"
      fi
    else
      echo "CUDA was not found, selecting CPU image. For GPU image, install NVIDIA-DOCKER, CUDA and NVCC."
    fi
  else
    from_image="nvidia/cuda:${docker_cuda}-cudnn${docker_cudnn}-devel-ubuntu16.04"
    image_label="espnet:cuda${docker_cuda}-cudnn${docker_cudnn}-ubuntu16.04"
  fi
fi
echo "Using image ${from_image}."
docker_image=$( docker images -q ${image_label} ) 
cd ..

if ! [[ -n ${docker_image}  ]]; then
  echo "Building docker image..."
  build_args="--build-arg FROM_IMAGE=${from_image}"
  if [ ! -z "${HTTP_PROXY}" ]; then
    echo "Building with proxy ${HTTP_PROXY}"
    build_args="${build_args} --build-arg WITH_PROXY=${HTTP_PROXY}"
  fi
  if [ ${docker_user} -gt 0 ]; then
    build_args="${build_args} --build-arg THIS_USER=${HOME##*/}"
    build_args="${build_args} --build-arg THIS_UID=${UID}"
  else
    build_args="${build_args} --build-arg THIS_USER=root"
  fi
  echo "Now running docker build ${build_args} -f docker/espnet.devel -t ${image_label} ."
  (docker build ${build_args} -f docker/espnet.devel -t ${image_label} .) || exit 1
fi

vols="-v ${PWD}/egs:/espnet/egs -v ${PWD}/src:/espnet/src -v ${PWD}/test:/espnet/test"
if [ ! -z "${docker_folders}" ]; then
  docker_folders=$(echo ${docker_folders} | tr "," "\n")
  for i in ${docker_folders[@]}
  do
    vols=${vols}" -v $i:$i";
  done
fi

# Test if link to kaldi_io.py has been created
if ! [[ -L ./src/utils/kaldi_io_py.py ]]; then
  my_dir=${PWD}
  cd ./src/utils
  ln -s ../../tools/kaldi-io-for-python/kaldi_io.py kaldi_io_py.py
  cd ${my_dir}
fi

cmd1="cd /espnet/egs/${docker_egs}"
cmd2="./run.sh $@"
# Required to access to the folder once the training if finished
cmd3="chmod -R 777 /espnet/egs/${docker_egs}"

cmd="${cmd1}; ${cmd2}; ${cmd3}"
if [ "${docker_gpu}" == "-1" ]; then
  cmd="docker run -i --rm --name espnet_cpu ${vols} ${image_label} /bin/bash -c '${cmd}'"
else
  # --rm erase the container when the training is finished.
  container_gpu=${docker_gpu//,/_}
  cmd="NV_GPU='${docker_gpu}' nvidia-docker run -i --rm --name espnet_gpu${container_gpu} ${vols} ${image_label} /bin/bash -c '${cmd}'"
fi

echo "Executing application in Docker"
echo ${cmd}
eval ${cmd}

echo "`basename $0` done."
