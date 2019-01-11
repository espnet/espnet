#!/bin/bash

this_path=`pwd`
stage=0
stop_stage=100
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Build docker containers"
    # build runtime and gpu based containers
    docker build -f prebuilt/runtime/Dockerfile -t espnet/espnet:runtime . || exit 1
    for ver in 8.0 9.0 9.1 9.2; do
        docker build -f prebuilt/devel/gpu/${ver}/cudnn7/Dockerfile -t espnet/espnet:cuda${ver}-cudnn7 . || exit 1
    done

    # build cpu based
    docker build --build-arg FROM_TAG=runtime -f prebuilt/devel/Dockerfile -t espnet/espnet:cpu . || exit 1

    # build gpu based
    for ver in 8.0 9.0 9.1 9.2; do
        docker build --build-arg FROM_TAG=cuda${ver}-cudnn7 -f prebuilt/devel/Dockerfile -t espnet/espnet:gpu-cuda${ver}-cudnn7 . || exit 1
    done
fi

tags="runtime
    cuda8.0-cudnn7
    cuda9.0-cudnn7
    cuda9.1-cudnn7
    cuda9.2-cudnn7
    cuda9.2-cudnn7
    cpu
    gpu-cuda8.0-cudnn7
    gpu-cuda9.0-cudnn7
    gpu-cuda9.1-cudnn7
    gpu-cuda9.2-cudnn7"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for tag in ${tags};do
        echo "docker push espnet/espnet:${tag}"
        ( docker push espnet/espnet:${tag} )|| exit 1
    done
fi

echo "`basename $0` done."