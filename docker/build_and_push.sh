#!/bin/bash

this_path=`pwd`
stage=0
if [ ${stage} -le 0 ]; then
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
fi

#Test Docker Containers with cpu setup
if [ ${stage} -le 1 ]; then
    for cuda_ver in cpu 8.0 9.0 9.1 9.2 10.0;do
        for backend in pytorch chainer;do
            if [ "${cuda_ver}" != "cpu" ];then
                docker_cuda="--docker_cuda ${cuda_ver}"
                gpu=0
                ngpu=1
                run_stage=3
            else
                docker_cuda=""
                gpu=-1
                ngpu=0
                run_stage=-1
            fi
            ( ./run.sh --docker_egs an4/asr1 ${docker_cuda} --docker_cmd run.sh --docker_gpu ${gpu} \
                        --verbose 1 --backend ${backend} --ngpu ${ngpu} \
                        --stage ${run_stage} --tag train_nodev_${backend}_cuda${cuda_ver}) || exit 1
        done
    done
fi


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

if [ ${stage} -le 2 ]; then
    for tag in ${tags};do
        echo "docker push espnet/espnet:${tag}"
        ( docker push espnet/espnet:${tag} )|| exit 1
    done
fi

echo "`basename $0` done."