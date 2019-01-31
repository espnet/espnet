PYTHON := /usr/bin/python2.7
TH_VERSION := 1.0.0  # 0.4.1 or 1.0.0
KALDI =
GCC_VERSION := $(shell gcc -dumpversion)

.PHONY: all clean

all: venv chainer_patch.done kaldi.done nkf.done warp-ctc.done chainer_ctc.done sentencepiece.done kaldiio pytorch_complex moses

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

venv:
	test -d venv || virtualenv -p $(PYTHON) venv

espnet.done: venv
	. venv/bin/activate; pip install pip --upgrade
	. venv/bin/activate; pip install -e ..
	. venv/bin/activate; pip install cupy==5.0.0 torch==$(TH_VERSION) matplotlib
	touch espnet.done

kaldiio: venv
	. venv/bin/activate; pip install git+https://github.com/nttcslab-sp/kaldiio.git

pytorch_complex: venv
	# Require python>=3.6, pytorch>=1.0
	. venv/bin/activate; pip install git+https://github.com/kamo-naoyuki/pytorch_complex

chainer_patch.done: espnet.done
	$(eval FILENAME := $(shell find venv/lib -name "multiprocess_iterator.py"))
	patch $(FILENAME) < prefetch.patch
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
	# Note(kamo): Requires gcc>=4.9 to build extensions with pytorch>=1.0
	if . venv/bin/activate && python -c 'import torch as t;assert t.__version__[0] == "1"' &> /dev/null; then \
        . venv/bin/activate && python -c "from distutils.version import LooseVersion as V;assert V('$(GCC_VERSION)') >= V('4.9'), 'Requires gcc>=4.9'"; \
	fi
	if . venv/bin/activate && python -c 'import torch as t;assert t.__version__[0] == "1"' &> /dev/null; then \
        cd warp-ctc; git checkout -b pytorch-1.0.0 remotes/origin/pytorch-1.0.0; \
    fi
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
	rm -fr kaldi venv nkf warp-ctc chainer_ctc sentencepiece moses
	rm -f miniconda.sh
	rm -rf *.done
	find . -iname "*.pyc" -delete

clean_python:
	rm -fr venv warp-ctc chainer_ctc kaldi-io-for-python
	rm -f miniconda.sh
	rm -rf chainer_patch.done warp-ctc.done chainer_ctc.done
	find . -iname "*.pyc" -delete
