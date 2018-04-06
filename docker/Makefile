.PHONY: pytorch chainer clean

chainer: requirements kaldi nkf kaldi-io-for-python chainer_ctc subword-nmt

pytorch: requirements kaldi nkf kaldi-io-for-python warp-ctc pytorch-src subword-nmt

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
requirements: requirements.txt
	pip install pip --upgrade; pip install -r requirements.txt; pip install matplotlib

nkf:
	mkdir -p nkf
	cd nkf; wget http://gigenet.dl.osdn.jp/nkf/64158/nkf-2.1.4.tar.gz
	cd nkf; tar zxvf nkf-2.1.4.tar.gz; cd nkf-2.1.4; $(MAKE) prefix=.

pytorch-src: 
	pip install pip --upgrade; pip install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl

warp-ctc: pytorch
	git clone https://github.com/SeanNaren/warp-ctc.git
	cd warp-ctc && git checkout 9e5b238f8d9337b0c39b3fd01bbaff98ba523aa5 && mkdir build && cd build && cmake .. && make -j4
	pip install cffi && 
	cd warp-ctc/pytorch_binding && python setup.py install # maybe need to: apt-get install python-dev

chainer_ctc: 
	git clone https://github.com/jheymann85/chainer_ctc.git
	pip install cython
	cd chainer_ctc && chmod +x install_warp-ctc.sh && ./install_warp-ctc.sh
	cd chainer_ctc && pip install .

subword-nmt:
	git clone https://github.com/rsennrich/subword-nmt.git

clean:
	rm -fr kaldi_github kaldi kaldi_python nkf kaldi-io-for-python ../src/utils/kaldi_io_py.py warp-ctc chainer_ctc subword-nmt
	rm -f miniconda.sh
	find . -iname "*.pyc" -delete
