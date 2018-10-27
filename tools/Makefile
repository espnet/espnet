PYTHON := /usr/bin/python2.7
KALDI =

.PHONY: all clean

all: venv chainer_patch.done kaldi nkf.done kaldi-io-for-python.done warp-ctc.done chainer_ctc.done sentencepiece.done moses

all_python: venv chainer_patch.done warp-ctc.done chainer_ctc.done

ifneq ($(strip $(KALDI)),)
kaldi:
	ln -s $(KALDI) kaldi
else
kaldi:
	test -d kaldi_github || git clone https://github.com/kaldi-asr/kaldi.git kaldi_github
	cd kaldi_github/tools; $(MAKE) all
	cd kaldi_github/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
	ln -nfs kaldi_github kaldi
endif

# for some reason, including matplotlib in requirements.txt causes errors, and
# matplotlib should be separately pip done
venv:
	test -d venv || venv -p $(PYTHON) venv

espnet.done: venv
	. venv/bin/activate; pip install pip --upgrade
	. venv/bin/activate; pip install -e ..
	. venv/bin/activate; pip install cupy==4.3.0 matplotlib
	touch espnet.done

kaldi-io-for-python.done: venv
	rm -rf kaldi-io-for-python
	git clone https://github.com/vesis84/kaldi-io-for-python.git
	cd kaldi-io-for-python; ln -s kaldi_io.py kaldi_io_py.py; cp ../setup_kaldi-io-for-python.py setup.py; pip install .
	touch kaldi-io-for-python.done

chainer_patch.done: espnet.done
	$(eval FILENAME := $(shell find venv/lib -name "multiprocess_iterator.py"))
	. venv/bin/activate; patch $(FILENAME) < prefetch.patch
	touch chainer_patch.done

nkf.done:
	rm -rf nkf
	mkdir -p nkf
	cd nkf; wget https://ja.osdn.net/dl/nkf/nkf-2.1.4.tar.gz
	cd nkf; tar zxvf nkf-2.1.4.tar.gz; cd nkf-2.1.4; $(MAKE) prefix=.
	touch nkf.done

warp-ctc.done: espnet.done
	rm -rf warp-ctc
	git clone https://github.com/jnishi/warp-ctc.git
	. venv/bin/activate; cd warp-ctc && mkdir build && cd build && cmake .. && make -j4 ; true
	. venv/bin/activate; pip install cffi
	. venv/bin/activate; cd warp-ctc/pytorch_binding && python setup.py install # maybe need to: apt-get install python-dev
	touch warp-ctc.done

chainer_ctc.done: espnet.done
	rm -rf chainer_ctc
	git clone https://github.com/jheymann85/chainer_ctc.git
	. venv/bin/activate; pip install cython
	. venv/bin/activate; cd chainer_ctc && chmod +x install_warp-ctc.sh && ./install_warp-ctc.sh ; true
	. venv/bin/activate; cd chainer_ctc && pip install .
	touch chainer_ctc.done

sentencepiece.done:
	rm -rf sentencepiece
	git clone https://github.com/google/sentencepiece.git
	cd sentencepiece && mkdir build && cd build && (cmake3 .. || cmake ..) && $(MAKE)
	touch sentencepiece.done

moses:
	git clone https://github.com/moses-smt/mosesdecoder.git moses

clean:
	rm -fr kaldi_github kaldi venv nkf kaldi-io-for-python warp-ctc chainer_ctc sentencepiece moses
	rm -f miniconda.sh
	rm -rf *.done
	find . -iname "*.pyc" -delete

clean_python:
	rm -fr venv warp-ctc chainer_ctc kaldi-io-for-python
	rm -f miniconda.sh
	rm -rf *.done
	find . -iname "*.pyc" -delete
