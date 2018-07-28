KALDI =

.PHONY: all clean

all: venv/bin/activate kaldi nkf kaldi-io-for-python venv/lib/python2.7/site-packages/torch warp-ctc chainer_ctc sentencepiece

kaldi-io-for-python:
	git clone https://github.com/vesis84/kaldi-io-for-python.git
	cd ../src/utils; ln -s ../../tools/kaldi-io-for-python/kaldi_io.py kaldi_io_py.py

ifneq ($(strip $(KALDI)),)
kaldi:
	ln -s $(KALDI) kaldi
else
kaldi:
	git clone https://github.com/kaldi-asr/kaldi.git kaldi_github; cd kaldi_github/tools; $(MAKE) all
	cd kaldi_github/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
	ln -s kaldi_github kaldi	
endif

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
	. venv/bin/activate; pip install pip --upgrade; pip install torch==0.4.0

warp-ctc: venv/lib/python2.7/site-packages/torch
	git clone https://github.com/jnishi/warp-ctc.git
	. venv/bin/activate; cd warp-ctc && mkdir build && cd build && cmake .. && make -j4 ; true
	. venv/bin/activate; pip install cffi
	. venv/bin/activate; cd warp-ctc/pytorch_binding && python setup.py install # maybe need to: apt-get install python-dev

chainer_ctc: venv/bin/activate
	git clone https://github.com/jheymann85/chainer_ctc.git
	. venv/bin/activate; pip install cython
	. venv/bin/activate; cd chainer_ctc && chmod +x install_warp-ctc.sh && ./install_warp-ctc.sh ; true
	. venv/bin/activate; cd chainer_ctc && pip install .

sentencepiece:
	git clone https://github.com/google/sentencepiece.git
	cd sentencepiece && mkdir build && cd build && (cmake3 .. || cmake ..) && $(MAKE)

clean:
	rm -fr kaldi_github kaldi kaldi_python venv nkf kaldi-io-for-python ../src/utils/kaldi_io_py.py warp-ctc chainer_ctc sentencepiece
	rm -f miniconda.sh
	find . -iname "*.pyc" -delete
