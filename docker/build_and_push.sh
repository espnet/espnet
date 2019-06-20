#!/bin/bash

this_path=`pwd`
stage=1
cuda_vers="10.0"
docker_ver=$(docker version -f '{{.Server.Version}}')

echo "Using Docker Ver.${docker_ver}"
if [ ${stage} -le 0 ]; then
    echo "Build docker containers"
    # build runtime and gpu based containers
    docker build --build-arg DOCKER_VER=${docker_ver} -f prebuilt/runtime/Dockerfile -t espnet/espnet:runtime . || exit 1
    for ver in ${cuda_vers}; do
        docker build -f prebuilt/devel/gpu/${ver}/cudnn7/Dockerfile -t espnet/espnet:cuda${ver}-cudnn7 . || exit 1
    done

    # build cpu based
    docker build --build-arg FROM_TAG=runtime -f prebuilt/devel/Dockerfile -t espnet/espnet:cpu-u18 . || exit 1

    # build gpu based
    for ver in ${cuda_vers}; do
        build_args="--build-arg FROM_TAG=cuda${ver}-cudnn7"
        build_args="${build_args} --build-arg CUDA_VER=${ver}"

        docker build ${build_args} -f prebuilt/devel/Dockerfile -t espnet/espnet:gpu-cuda${ver}-cudnn7-u18 . || exit 1
    done
fi

#Test Docker Containers with cpu setup
if [ ${stage} -le 1 ]; then
    run_stage=-1
    for cuda_ver in cpu ${cuda_vers};do    
        for backend in pytorch chainer;do
            if [ "${cuda_ver}" != "cpu" ];then
                docker_cuda="--docker_cuda ${cuda_ver}"
                gpu=0
                ngpu=1
            else
                docker_cuda=""
                gpu=-1
                ngpu=0
            fi
            ( ./run.sh --docker_egs an4/asr1 ${docker_cuda} --docker_cmd run.sh --docker_gpu ${gpu} \
                        --verbose 1 --backend ${backend} --ngpu ${ngpu} \
                        --stage ${run_stage} --tag train_nodev_${backend}_cuda${cuda_ver}) || exit 1
            run_stage=3
        done
    done
fi

exit 1
tags="cpu
      gpu-cuda9.2-cudnn7
      gpu-cuda10.0-cudnn7"

if [ ${stage} -le 2 ]; then
    for tag in ${tags};do
        echo "docker push espnet/espnet:${tag}"
        ( docker push espnet/espnet:${tag} )|| exit 1
    done
fi

echo "`basename $0` done."
