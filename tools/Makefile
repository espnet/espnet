# If a Python interpreter is specified, then creates a virtualenv from it
# PYTHON := /usr/bin/python2.7
PYTHON :=
# The python version installed in the conda setup
PYTHON_VERSION := 3.7
CUPY_VERSION := 5.0.0
# PyTorch version: 0.4.1 or 1.0.0
TH_VERSION := 1.0.0
# Use a prebuild Kaldi to omit the installation
KALDI :=

# Both Miniconda2/3 can install any Python versions
CONDA_URL := https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# PyTorch>=1.0.0 requires gcc>=4.9 when buliding the extensions
GCC_VERSION := $(shell gcc -dumpversion)

.PHONY: all clean

all: venv chainer_patch.done kaldi.done nkf.done warp-ctc.done chainer_ctc.done sentencepiece.done kaldiio moses

all_python: venv chainer_patch.done warp-ctc.done chainer_ctc.done


ifneq ($(strip $(KALDI)),)
kaldi.done:
	ln -s $(abspath $(KALDI)) kaldi
	touch kaldi.done
else
kaldi.done:
	test -d kaldi || git clone https://github.com/kaldi-asr/kaldi.git
	cd kaldi/tools; $(MAKE) all
	cd kaldi/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
	touch kaldi.done
endif


ifneq ($(strip $(PYTHON)),)
venv:
	test -d venv || virtualenv -p $(PYTHON) venv
espnet.done: venv
	. venv/bin/activate; pip install pip --upgrade
	. venv/bin/activate; pip install -e ..
	. venv/bin/activate; pip install cupy==$(CUPY_VERSION) torch==$(TH_VERSION) matplotlib
	touch espnet.done
else
miniconda.sh:
	test -f miniconda.sh || wget $(CONDA_URL) -O miniconda.sh
venv: miniconda.sh
	test -d $(PWD)/venv || bash miniconda.sh -b -p $(PWD)/venv
	. venv/bin/activate && conda update -y conda
	. venv/bin/activate && conda install -y python=$(PYTHON_VERSION)
	. venv/bin/activate && conda info -a
espnet.done: venv
	. venv/bin/activate && conda install -y pytorch=$(TH_VERSION) -c pytorch
	. venv/bin/activate && conda install -y matplotlib
	. venv/bin/activate && pip install -e ..
	. venv/bin/activate && pip install cupy==$(CUPY_VERSION) 
	touch espnet.done
endif

kaldiio: venv
	. venv/bin/activate; pip install git+https://github.com/nttcslab-sp/kaldiio.git

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
