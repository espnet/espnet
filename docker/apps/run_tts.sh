#!/bin/bash


is_root=false

if [ $# != 1 ]; then
    echo "Wrong #arguments ($#, expected 1)"
    echo "Usage: apps/run_tts.sh [options] <text>"
    exit 1
fi

TTS_TEXT=$1

cd ..

from_tag=cpu-u18
EXTRAS=true

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

        echo "Now running docker build ${build_args} -f prebuilt/Dockerfile -t espnet/espnet:${container_tag} ."
        (docker build ${build_args} -f prebuilt/Dockerfile -t  espnet/espnet:${container_tag} .) || exit 1
    fi
else
    container_tag=${from_tag}
fi

cd ..


vols="-v ${PWD}/egs:/espnet/egs
      -v ${PWD}/espnet:/espnet/espnet
      -v ${PWD}/test:/espnet/test 
      -v ${PWD}/utils:/espnet/utils"


docker run --rm ${vols} espnet/espnet:${container_tag} bash -c "cd espnet/egs/ljspeech/tts1;
                                mkdir -p data;
                                echo '${TTS_TEXT}' > data/text;
                                /espnet/utils/synth_wav.sh data/text"
