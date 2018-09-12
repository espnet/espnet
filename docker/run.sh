#!/bin/bash

docker_gpu=0
docker_egs=
docker_folders=
docker_cuda=9.1
docker_user=true
docker_env=


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

from_tag="cpu"
if [ ! "${docker_gpu}" == "-1" ]; then
  if [ -z "${docker_cuda}" ]; then
    # If the docker_cuda is not set, the program will automatically 
    # search the installed version with default configurations (apt)
    docker_cuda=$( nvcc -V | grep release )
    docker_cuda=${docker_cuda#*"release "}
    docker_cuda=${docker_cuda%,*}
  fi
  # After search for your cuda version, if the variable docker_cuda is empty the program will raise an error
  if [ -z "${docker_cuda}" ]; then
    echo "CUDA was not found in your system. Use CPU image or install NVIDIA-DOCKER, CUDA and NVCC for GPU image."
    exit 1
  else
    from_tag="gpu-cuda${docker_cuda}-cudnn7"
  fi
fi

# Check if image exists in the system and download if required
docker_image=$( docker images -q espnet/espnet:${from_tag} )
if ! [[ -n ${docker_image}  ]]; then
  docker pull espnet/espnet:${from_tag}
fi

if [ ${docker_user} ]; then
  container_tag="${from_tag}-user-${HOME##*/}"
else
  container_tag=${from_tag}
fi

echo "Using image espnet/espnet:${container_tag}."
docker_image=$( docker images -q espnet/espnet:${container_tag} ) 

this_time="$(date '+%Y%m%dT%H%M')"
if [ "${docker_gpu}" == "-1" ]; then
  cmd0="docker"
  container_name="espnet_cpu_${this_time}"
else
  # --rm erase the container when the training is finished.
  cmd0="NV_GPU='${docker_gpu}' nvidia-docker"
  container_name="espnet_gpu${docker_gpu//,/_}_${this_time}"
fi

if ! [[ -n ${docker_image}  ]]; then
  echo "Building docker image..."
  build_args="--build-arg FROM_TAG=${from_tag}"

  if [ ${docker_user} ]; then
    build_args="${build_args} --build-arg THIS_USER=${HOME##*/}"
    build_args="${build_args} --build-arg THIS_UID=${UID}"
  else
    build_args="${build_args} --build-arg THIS_USER=root"
  fi
  echo "Now running docker build ${build_args} -f prebuilt/Dockerfile -t espnet/espnet:${container_tag} ."
  (docker build ${build_args} -f prebuilt/Dockerfile -t  espnet/espnet:${container_tag} .) || exit 1
fi

cd ..

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
if [ ${docker_user} ]; then
  cmd2="./run.sh $@"
else
  # Required to access to the folder once the training if finished
  cmd2="${cmd2}; chmod -R 777 /espnet/egs/${docker_egs}"
fi

cmd="${cmd1}; ${cmd2}"
this_env=""
if [ ! -z "${docker_env}" ]; then
  docker_env=$(echo ${docker_env} | tr "," "\n")
  for i in ${docker_env[@]}
  do
    this_env="-e $i ${this_env}" 
  done
fi

if [ ! -z "${HTTP_PROXY}" ]; then
  this_env="${this_env} -e 'HTTP_PROXY=${HTTP_PROXY}'"
fi

if [ ! -z "${http_proxy}" ]; then
  this_env="${this_env} -e 'http_proxy=${http_proxy}'"
fi

cmd="${cmd0} run -i --rm ${this_env} --name ${container_name} ${vols} espnet/espnet:${container_tag} /bin/bash -c '${cmd}'"

trap ctrl_c INT

function ctrl_c() {
        echo "** Kill docker container ${container_name}"
        docker rm -f ${container_name}
}

echo "Executing application in Docker"
echo ${cmd}
eval ${cmd} &
PROC_ID=$!

while kill -0 "$PROC_ID" 2> /dev/null; do
    sleep 1
done
echo "`basename $0` done."
