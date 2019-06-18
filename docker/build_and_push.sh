#!/bin/bash


SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

tags="runtime
    cuda8.0-cudnn7
    cuda9.0-cudnn7
    cuda9.1-cudnn7
    cuda9.2-cudnn7
    cuda9.2-cudnn7
    cuda10.0-cudnn7
    cpu
    gpu-cuda8.0-cudnn7
    gpu-cuda9.0-cudnn7
    gpu-cuda9.1-cudnn7
    gpu-cuda9.2-cudnn7
    gpu-cuda10.0-cudnn7"


cmd_usage() {
    PROGRAM=$(basename "$0")
    cat >&2 <<-_EOF
    DESCRIPTION
        A script for automatic builds of docker images for ESPnet and
        pushing them to Dockerhub.
        Also able to build containers based on local build configuration.
    
    USAGE
        ${PROGRAM} <mode>
        ${PROGRAM} build_and_push
        ${PROGRAM} local [cpu|9.1|9.2|10.0|10.1]

            mode      Select script functionality

        Modes
            build           build docker containers
            test            test docker containers
            push            push to docker hub
            build_and_push  automated build, test and push
            local           build a docker container from the local ESPnet repository
                            using the base image from docker hub (espnet/espnet:runtime)
                            optional: cpu or CUDA version (default: cpu)
            fully_local     like local, but also build the base image

	_EOF
    exit 1
}


build(){
    echo "Build docker containers"
    # build runtime and gpu based containers
    docker build -f prebuilt/runtime/Dockerfile -t espnet/espnet:runtime . || exit 1
    for ver in 8.0 9.0 9.1 9.2 10.0; do
        docker build -f prebuilt/devel/gpu/${ver}/cudnn7/Dockerfile -t espnet/espnet:cuda${ver}-cudnn7 . || exit 1
    done

    # build cpu based
    docker build --build-arg FROM_TAG=runtime -f prebuilt/devel/Dockerfile -t espnet/espnet:cpu . || exit 1

    # build gpu based
    for ver in 8.0 9.0 9.1 9.2 10.0; do
        docker build --build-arg FROM_TAG=cuda${ver}-cudnn7 -f prebuilt/devel/Dockerfile -t espnet/espnet:gpu-cuda${ver}-cudnn7 . || exit 1
    done
}


build_local(){
    echo "Building docker container: base image, and image for ${ver}"
    sleep 1

    # prepare espnet-repo, assuming that this script is in folder espnet/docker
    cd ${SCRIPTPATH}/..
    ESPNET_ARCHIVE="./espnet-local.tar"
    echo "Reconstructing the local repository from the last commit"
    git archive -o docker/${ESPNET_ARCHIVE} HEAD  || exit 1
    cd ${SCRIPTPATH}
    test -r ${ESPNET_ARCHIVE} || exit 1;
    sleep 1

    if [ "${build_base_image}" = true ] ; then
        echo "building ESPnet base image"
        docker build -f prebuilt/runtime/Dockerfile -t espnet/espnet:runtime . || exit 1
        sleep 1
    fi

    if [[ ${ver} == "cpu" ]]; then
        echo "building ESPnet CPU Image"
        docker build --build-arg FROM_TAG=runtime  --build-arg ESPNET_ARCHIVE=${ESPNET_ARCHIVE} \
                     -f prebuilt/local/Dockerfile -t espnet/espnet:cpu .
    elif [[ ${ver} =~ ^(9.1|9.2|10.0|10.1)$ ]]; then
            echo "building ESPnet GPU Image for ${ver}"
        docker build -f prebuilt/devel/gpu/${ver}/cudnn7/Dockerfile -t espnet/espnet:cuda${ver}-cudnn7 .
        docker build --build-arg FROM_TAG=cuda${ver}-cudnn7 --build-arg ESPNET_ARCHIVE=${ESPNET_ARCHIVE} \
                     -f prebuilt/local/Dockerfile -t espnet/espnet:gpu-cuda${ver}-cudnn7 .
    else
        echo "Parameter invalid: " ${ver}
    fi

    echo "cleanup."
    test -r ${ESPNET_ARCHIVE} && rm ${ESPNET_ARCHIVE}
}


testing(){
    echo "Testing docker containers"
    # Test Docker Containers with cpu setup
    run_stage=-1
    for cuda_ver in cpu 8.0 9.0 9.1 9.2 10.0;do
        for backend in pytorch chainer;do
            docker_cuda=""
            gpu=-1
            ngpu=0
            if [ "${cuda_ver}" != "cpu" ]; then
                docker_cuda="--docker_cuda ${cuda_ver}"
                gpu=2
                ngpu=1
            fi
            ( ./run.sh --docker_egs an4/asr1 ${docker_cuda} --docker_cmd run.sh --docker_gpu ${gpu} --verbose 1 --backend ${backend} --ngpu ${ngpu} --stage ${run_stage} ) || exit 1
            if [ ${run_stage} -eq -1 ]; then
                run_stage=3
            fi
        done
    done
}


push(){
    for tag in ${tags};do
        echo "docker push espnet/espnet:${tag}"
        ( docker push espnet/espnet:${tag} )|| exit 1
    done
}



## Parameter initialization: cpu or gpu docker container (default: cpu)
if [[ -z "$2" ]]; then
    ver='cpu'
else
    ver=$2
fi


## Application menu
if   [[ $1 == "build" ]]; then
    build
elif [[ $1 == "local" ]]; then
    build_base_image=false
    build_local
elif [[ $1 == "fully_local" ]]; then
    build_base_image=true
    build_local_gpu
elif [[ $1 == "push" ]]; then
    push
elif [[ $1 == "build_and_push" ]]; then
    build
    testing
    push
else
    cmd_usage
fi

echo "$(basename "$0") done."
