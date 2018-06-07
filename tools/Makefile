.PHONY: all clean

all: venv/bin/activate kaldi nkf ../src/utils/kaldi-io-for-python.py venv/lib/python2.7/site-packages/torch warp-ctc/build chainer_ctc/ext

../src/utils/kaldi-io-for-python.py:
	cd ../src/utils; ln -s ../../tools/kaldi-io-for-python/kaldi_io.py kaldi_io_py.py

kaldi:
	cd kaldi_github/tools; $(MAKE) all
	cd kaldi_github/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
	ln -s kaldi_github kaldi

# for some reason, including matplotlib in requirements.txt causes errors, and
# matplotlib should be separately pip installed
venv/bin/activate: requirements.txt
	test -d venv || virtualenv -p /usr/bin/python2.7 venv
	. venv/bin/activate; pip install pip --upgrade; pip install -r requirements.txt --no-cache; pip install matplotlib
	touch venv/bin/activate

nkf:
	mkdir -p nkf
	cd nkf; wget http://gigenet.dl.osdn.jp/nkf/64158/nkf-2.1.4.tar.gz
	cd nkf; tar zxvf nkf-2.1.4.tar.gz; cd nkf-2.1.4; $(MAKE) prefix=.

venv/lib/python2.7/site-packages/torch: venv/bin/activate
	. venv/bin/activate; pip install pip --upgrade; pip install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl

warp-ctc/build: venv/lib/python2.7/site-packages/torch
	. venv/bin/activate; cd warp-ctc  && mkdir build && cd build && cmake .. && $(MAKE) -j4
	. venv/bin/activate; cd warp-ctc/pytorch_binding && python setup.py install # maybe need to: apt-get install python-dev

chainer_ctc/ext: venv/bin/activate
	. venv/bin/activate; cd chainer_ctc && ./install_warp-ctc.sh
	. venv/bin/activate; cd chainer_ctc && pip install .

clean:
	cd kaldi_github/tools && $(MAKE) clean
	rm -fr kaldi kaldi_python venv nkf ../src/utils/kaldi_io_py.py warp-ctc/build chainer_ctc/ext
	rm -f miniconda.sh
	rm -f chainer300_cudnn7.0_nossh.devel
	find . -iname "*.pyc" -delete
