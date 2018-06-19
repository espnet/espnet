.PHONY: all clean

all: venv/bin/activate kaldi nkf kaldi-io-for-python venv/lib/python2.7/site-packages/torch warp-ctc chainer_ctc subword-nmt

kaldi-io-for-python:
	git clone https://github.com/vesis84/kaldi-io-for-python.git
	cd ../src/utils; ln -s ../../tools/kaldi-io-for-python/kaldi_io.py kaldi_io_py.py

kaldi: kaldi_github
	cd kaldi_github/tools; $(MAKE) all
	cd kaldi_github/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
	ln -s kaldi_github kaldi

kaldi_github:
	git clone https://github.com/kaldi-asr/kaldi.git kaldi_github

# for some reason, including matplotlib in requirements.txt causes errors, and
# matplotlib should be separately pip installed
venv/bin/activate: requirements.txt
	test -d venv || virtualenv -p /usr/bin/python2.7 venv
	. venv/bin/activate; pip install pip --upgrade; pip install -r requirements.txt; pip install matplotlib
	touch venv/bin/activate

nkf:
	mkdir -p nkf
	cd nkf; wget https://ja.osdn.net/dl/nkf/nkf-2.1.4.tar.gz
	cd nkf; tar zxvf nkf-2.1.4.tar.gz; cd nkf-2.1.4; $(MAKE) prefix=.

venv/lib/python2.7/site-packages/torch: venv/bin/activate
	. venv/bin/activate; pip install pip --upgrade; pip install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl

warp-ctc: venv/lib/python2.7/site-packages/torch
	git clone https://github.com/SeanNaren/warp-ctc.git
	. venv/bin/activate; cd warp-ctc && git checkout 9e5b238f8d9337b0c39b3fd01bbaff98ba523aa5 && mkdir build && cd build && cmake .. && make -j4
	. venv/bin/activate; pip install cffi
	. venv/bin/activate; cd warp-ctc/pytorch_binding && python setup.py install # maybe need to: apt-get install python-dev

chainer_ctc: venv/bin/activate
	git clone https://github.com/jheymann85/chainer_ctc.git
	. venv/bin/activate; pip install cython
	. venv/bin/activate; cd chainer_ctc && chmod +x install_warp-ctc.sh && ./install_warp-ctc.sh
	. venv/bin/activate; cd chainer_ctc && pip install .

subword-nmt:
	git clone https://github.com/rsennrich/subword-nmt.git

clean:
	rm -fr kaldi_github kaldi kaldi_python venv nkf kaldi-io-for-python ../src/utils/kaldi_io_py.py warp-ctc chainer_ctc subword-nmt
	rm -f miniconda.sh
	rm -f chainer300_cudnn7.0_nossh.devel
	find . -iname "*.pyc" -delete
