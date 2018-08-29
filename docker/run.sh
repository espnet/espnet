#!/bin/bash

docker_gpu=0
docker_egs=
docker_folders=
docker_cuda=
docker_cudnn=7
docker_user=0
docker_env=

. ./parser.sh || exit 1

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
  echo "Now running docker build ${build_args} -f ${docker_file} -t ${image_label} ."
  (docker build ${build_args} -f ${docker_file} -t ${image_label} .) || exit 1
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
this_env=""
if [ ! -z "${docker_env}" ]; then
  this_env="-e ${docker_env}"
fi

cmd="${cmd0} run -i --rm ${this_env} --name ${container_name} ${vols} ${image_label} /bin/bash -c '${cmd}'"

echo "Executing application in Docker"
echo ${cmd}
eval ${cmd}

echo "`basename $0` done."
