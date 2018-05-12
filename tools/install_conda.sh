#!/bin/bash


# upgrade pip
pip install pip --upgrade

# install tools in pip
pip install -r requirements_conda.txt
pip install matplotlib

# cache current folder
INSTALL_ROOT_FOLDER=$(pwd)

# install kaldi
git clone https://github.com/kaldi-asr/kaldi.git kaldi_github
cd kaldi_github/tools; make all -j
cd ../src; ./configure --shared --use-cuda=no; make depend -j; make all -j
cd $INSTALL_ROOT_FOLDER; ln -s kaldi_github kaldi

# install nkf
mkdir -p nkf
cd nkf; wget http://gigenet.dl.osdn.jp/nkf/64158/nkf-2.1.4.tar.gz
tar zxvf nkf-2.1.4.tar.gz; cd nkf-2.1.4; make prefix=. -j
cd $INSTALL_ROOT_FOLDER

# install kaldi-io-for-python
git clone https://github.com/vesis84/kaldi-io-for-python.git
cd ../src/utils; ln -s ../../tools/kaldi-io-for-python/kaldi_io.py kaldi_io_py.py
cd $INSTALL_ROOT_FOLDER

# install pytorch (cuda 8, python 2.7)
conda install --yes pytorch torchvision -c pytorch

# install warp-ctc
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc && git checkout 9e5b238f8d9337b0c39b3fd01bbaff98ba523aa5 && mkdir build && cd build && cmake .. && make -j
cd ../pytorch_binding && python setup.py install
cd $INSTALL_ROOT_FOLDER

# install chainer-ctc
git clone https://github.com/jheymann85/chainer_ctc.git
cd chainer_ctc && chmod +x install_warp-ctc.sh && ./install_warp-ctc.sh
pip install .
cd $INSTALL_ROOT_FOLDER

# install (download) subword-nmt
git clone https://github.com/rsennrich/subword-nmt.git
cd $INSTALL_ROOT_FOLDER
