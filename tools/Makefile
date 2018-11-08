PYTHON := /usr/bin/python2.7
KALDI =

.PHONY: all clean

all: venv chainer_patch.done kaldi.done nkf.done kaldi-io-for-python.done warp-ctc.done chainer_ctc.done sentencepiece.done moses nara_wpe

all_python: venv chainer_patch.done warp-ctc.done chainer_ctc.done

ifneq ($(strip $(KALDI)),)
kaldi.done:
	ln -s $(KALDI) kaldi
	touch kaldi.done
else
kaldi.done:
	test -d kaldi || git clone https://github.com/kaldi-asr/kaldi.git
	cd kaldi/tools; $(MAKE) all
	cd kaldi/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
	touch kaldi.done
endif

# for some reason, including matplotlib in requirements.txt causes errors, and
# matplotlib should be separately pip installed
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

nara_wpe: venv
	. venv/bin/activate; pip install git+https://github.com/fgnt/nara_wpe.git
	. venv/bin/activate; pip install soundfile

sentencepiece.done:
	rm -rf sentencepiece
	git clone https://github.com/google/sentencepiece.git
	cd sentencepiece && mkdir build && cd build && (cmake3 .. || cmake ..) && $(MAKE)
	touch sentencepiece.done

moses:
	git clone https://github.com/moses-smt/mosesdecoder.git moses

clean:
	rm -fr kaldi venv nkf kaldi-io-for-python warp-ctc chainer_ctc sentencepiece moses
	rm -f miniconda.sh
	rm -rf *.done
	find . -iname "*.pyc" -delete

clean_python:
	rm -fr venv warp-ctc chainer_ctc kaldi-io-for-python
	rm -f miniconda.sh
	rm -rf chainer_patch.done kaldi-io-for-python.done warp-ctc.done chainer_ctc.done
	find . -iname "*.pyc" -delete
