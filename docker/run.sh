#!/usr/bin/env bash

docker_gpu=0
docker_egs=
docker_folders=
docker_tag=latest

docker_env=
docker_cmd=

is_root=false
is_local=false
is_egs2=false
is_extras=false

while test $# -gt 0
do
    case "$1" in
        -h) echo "Usage: `basename $0` [-h] docker_gpu docker_egs docker_folders options"
            exit 0;;
        --help) echo "Usage: `basename $0` [-h] ] docker_gpu docker_egs docker_folders options"
            exit 0;;
        --docker*) ext=${1#--}
            ext=${ext//-/_}
            frombreak=true
            for i in _ {a..z} {A..Z}; do
                for var in `eval echo "\\${!${i}@}"`; do
                    if [ "$var" == "$ext" ]; then
                        eval ${ext}=$2
                        frombreak=false
                        shift
                        break 2
                    fi
                done
            done
            if ${frombreak} ; then
                echo "bad option $1"
                exit 1
            fi
            ;;
        --is*) ext=${1#--}
            ext=${ext//-/_}
            frombreak=true
            for i in _ {a..z} {A..Z}; do
                for var in `eval echo "\\${!${i}@}"`; do
                    if [ "$var" == "$ext" ]; then
                        eval ${ext}=true
                        frombreak=false
                        break 2
                    fi
                done
            done
            if ${frombreak} ; then
                echo "bad option $1"
                exit 1
            fi
            ;;
        --*) break
            ;;
    esac
    shift
done

if [ -z "${docker_egs}" ]; then
    echo "Select an example to work with from the egs folder by setting --docker-egs."
    exit 1
fi

from_tag="cpu"
if [ ! "${docker_gpu}" == "-1" ]; then
    docker_cuda=$( nvcc -V | grep release )
    docker_cuda=${docker_cuda#*"release "}
    docker_cuda=${docker_cuda%,*}

    # After search for your cuda version, if the variable docker_cuda is empty the program will raise an error
    if [ -z "${docker_cuda}" ]; then
        echo "CUDA was not found in your system. Use CPU image or install NVIDIA-DOCKER, CUDA for GPU image."
        exit 1
    fi
        from_tag="gpu"
fi

from_tag="${from_tag}-${docker_tag}"

EXTRAS=${is_extras}

if [ ${is_local} = true ]; then
    from_tag="${from_tag}-local"
fi

# Check if image exists in the system and download if required
docker_image=$( docker images -q espnet/espnet:${from_tag} )
if ! [[ -n ${docker_image}  ]]; then
    if [ ${is_local} = true ]; then
        echo "!!! Warning: You need first to build the container using ./build.sh local <cuda_ver>."
        exit 1
    else
        docker pull espnet/espnet:${from_tag}
    fi
fi

if [ ${UID} -eq 0 ] && [ ${is_root} = false ]; then
    echo "Warning: Your user ID belongs to root users.
        Using Docker container with root instead of User-built container."
        is_root=true
fi

if [ ${is_root} = false ]; then
    # Build a container with the user account
    container_tag="${from_tag}-user-${HOME##*/}"
    docker_image=$( docker images -q espnet/espnet:${container_tag} )
    if ! [[ -n ${docker_image}  ]]; then
        echo "Building docker image..."
        build_args="--build-arg FROM_TAG=${from_tag}"
        build_args="${build_args} --build-arg THIS_USER=${HOME##*/}"
        build_args="${build_args} --build-arg THIS_UID=${UID}"
        build_args="${build_args} --build-arg EXTRA_LIBS=${EXTRAS}"

        echo "Now running docker build ${build_args} -f espnet.dockerfile -t espnet/espnet:${container_tag} ."
        (docker build ${build_args} -f espnet.dockerfile -t  espnet/espnet:${container_tag} .) || exit 1
    fi
else
    container_tag=${from_tag}
fi

echo "Using image espnet/espnet:${container_tag}."

this_time="$(date '+%Y%m%dT%H%M')"
if [ "${docker_gpu}" == "-1" ]; then
    cmd0="docker run "
    container_name="espnet_cpu_${this_time}"
else
    # --rm erase the container when the training is finished.
    if [ -z "$( which nvidia-docker )" ]; then
        # we assume that you already installed nvidia-docker 2
        cmd0="docker run --gpus '\"device=${docker_gpu}\"'"
    else
        cmd0="NV_GPU='${docker_gpu}' nvidia-docker run "
    fi
    container_name="espnet_gpu${docker_gpu//,/_}_${this_time}"
fi

cd ..

vols="-v ${PWD}/egs:/espnet/egs
      -v ${PWD}/espnet:/espnet/espnet
      -v ${PWD}/test:/espnet/test
      -v ${PWD}/utils:/espnet/utils"

in_egs=egs
if [ ${is_egs2} = true ]; then
    vols="${vols}   -v ${PWD}/egs2:/espnet/egs2
                    -v ${PWD}/espnet2:/espnet/espnet2
                    -v /dev/shm:/dev/shm"
    in_egs=egs2
fi

if [ ! -z "${docker_folders}" ]; then
    docker_folders=$(echo ${docker_folders} | tr "," "\n")
    for i in ${docker_folders[@]}
    do
        vols=${vols}" -v $i:$i";
    done
fi

cmd1="cd /espnet/${in_egs}/${docker_egs}"
if [ ! -z "${docker_cmd}" ]; then
    cmd2="./${docker_cmd} $@"
else
    cmd2="./run.sh $@"
fi

if [ ${is_root} = true ]; then
    # Required to access to the folder once the training if finished in root access
    cmd2="${cmd2}; chmod -R 777 /espnet/${in_egs}/${docker_egs}"
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

cmd="${cmd0} -i --rm ${this_env} --name ${container_name} ${vols} espnet/espnet:${container_tag} /bin/bash -c '${cmd}'"

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
