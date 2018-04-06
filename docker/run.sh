#!/bin/bash

gpu=0
backend=chainer
egs=voxforge
stage=0
corpus_dir=
while test $# -gt 0
do
    case "$1" in
        -h) echo "Usage: `basename $0` [-h] gpu model rotation type"
            exit 0;;
        --help) echo "Usage: `basename $0` [-h] gpu model rotation type"
              exit 0;;
        --*) ext=${1#--}
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
        *) echo "argument $1 does not exit"
            exit 1;;
    esac
    shift
    shift
done

name="ch300_u1604_c90_cdnn7_nossh"
image="espnet_chainer/3.x:9.0-cudnn7-16.04"

docker_image=$( docker images -q $image ) 

if ! [[ -n $docker_image  ]]; then
  echo "Building docker image..."
  (docker build -f "$name".devel -t $image .) || exit 1
fi

vol1="$PWD/../src:/espnet/src"
vol2="$PWD/../egs:/espnet/egs"
vol3="$PWD/../test:/espnet/test"
vol4="$corpus_dir:/$egs"

cmd0="cd /espnet/src/utils; ln -s /espnet/tools/kaldi-io-for-python/kaldi_io.py kaldi_io_py.py"
cmd1="cd /espnet/egs/$egs/asr1"
cmd2="./run.sh --gpu 0 --stage $stage"
cmd3="chmod -R 777 /espnet/egs/$egs/asr1" #Required to access once the training if finished

if [ ${gpu} -le -1 ]; then
  cmd="docker run -i --rm --name spnet_nogpu -v $vol1 -v $vol2 -v $vol3 $image /bin/bash -c '$cmd1; $cmd2; $cmd3'"
else
  cmd="NV_GPU=$gpu nvidia-docker run -i --rm --name spnet$gpu -v $vol1 -v $vol2 -v $vol3 -v $vol4 $image /bin/bash -c '$cmd0; $cmd1; $cmd2; $cmd3'" 
	# --rm erase the container when the training is finished.
fi

echo "Executing application in Docker"
echo $cmd
eval $cmd


echo "`basename $0` done."
