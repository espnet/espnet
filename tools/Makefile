# If a Python interpreter is specified, then creates a virtualenv from it
# PYTHON := /usr/bin/python3.7
PYTHON :=
# The python version installed in the conda setup
# NOTE(kan-bayashi): Use 3.7.3 to avoid sentencepiece installation error
PYTHON_VERSION := 3.7.3
CUPY_VERSION := 6.0.0
CUDA_VERSION := 10.0
# PyTorch version: 0.4.1 or 1.0.0 or 1.0.1
TH_VERSION := 1.0.1
# Use a prebuild Kaldi to omit the installation
KALDI :=
WGET := wget --tries=3

# Both Miniconda2/3 can install any Python versions
CONDA_URL := https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# PyTorch>=1.0.0 requires gcc>=4.9 when buliding the extensions
GCC_VERSION := $(shell gcc -dumpversion)


ifneq ($(shell which nvidia-smi),) # 'nvcc' found
CONDA_PYTORCH := pytorch=$(TH_VERSION) cudatoolkit=$(CUDA_VERSION)
CUDA_DEPS := cupy.done
else
CONDA_PYTORCH := pytorch-cpu=$(TH_VERSION)
CUDA_DEPS :=
endif


.PHONY: all clean

all: kaldi.done python check_install

python: venv $(CUDA_DEPS) warp-ctc.done warp-transducer.done chainer_ctc.done

extra: nkf.done moses.done mwerSegmenter.done pesq

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
	. venv/bin/activate; pip install torch==$(TH_VERSION)
	touch espnet.done
else
miniconda.sh:
	test -f miniconda.sh || $(WGET) $(CONDA_URL) -O miniconda.sh
venv: miniconda.sh
	test -d $(PWD)/venv || bash miniconda.sh -b -p $(PWD)/venv
	. venv/bin/activate && conda install -y setuptools -c anaconda
	. venv/bin/activate && conda install -y pip -c anaconda
	. venv/bin/activate && conda update -y conda
	. venv/bin/activate && conda install -y python=$(PYTHON_VERSION)
	. venv/bin/activate && conda info -a
espnet.done: venv
	. venv/bin/activate && conda install -y $(CONDA_PYTORCH) -c pytorch
	. venv/bin/activate && pip install -e ..
	touch espnet.done
endif

cupy.done: espnet.done
ifneq ($(strip $(CUPY_VERSION)),)
	. venv/bin/activate && pip install cupy-cuda$(strip $(subst .,,$(CUDA_VERSION)))==$(CUPY_VERSION)
	touch cupy.done
endif

warp-ctc.done: espnet.done
	if . venv/bin/activate && python -c 'import torch as t;major=t.__version__.split(".")[0];assert major == "1"' &> /dev/null; then \
		if [ ! -z "$(strip $(CUPY_VERSION))" ]; then \
			. venv/bin/activate && pip install warpctc-pytorch10-cuda$(strip $(subst .,,$(CUDA_VERSION))); \
		else \
			. venv/bin/activate && pip install warpctc-pytorch10-cpu; \
		fi \
	else \
		rm -rf warp-ctc; \
		git clone https://github.com/espnet/warp-ctc.git; \
		cd warp-ctc; git checkout -b pytorch-0.4 remotes/origin/pytorch-0.4; \
		mkdir build && cd build && cmake .. && $(MAKE) && cd ..; \
		. ../venv/bin/activate; pip install cffi; cd pytorch_binding && python setup.py install; \
	fi
	touch warp-ctc.done

warp-transducer.done: espnet.done
	rm -rf warp-transducer
	git clone https://github.com/HawkAaron/warp-transducer.git
	# Note: Requires gcc>=5.0 to build extensions with pytorch>=1.0
	if . venv/bin/activate && python -c 'import torch as t;assert t.__version__[0] == "1"' &> /dev/null; then \
        . venv/bin/activate && python -c "from distutils.version import LooseVersion as V;assert V('$(GCC_VERSION)') >= V('5.0'), 'Requires gcc>=5.0'"; \
	fi
	. venv/bin/activate; cd warp-transducer && mkdir build && cd build && cmake .. && $(MAKE); true
	. venv/bin/activate; cd warp-transducer/pytorch_binding && python setup.py install
	touch warp-transducer.done

chainer_ctc.done: espnet.done
	rm -rf chainer_ctc
	git clone https://github.com/jheymann85/chainer_ctc.git
	. venv/bin/activate; pip install cython
	. venv/bin/activate; cd chainer_ctc && chmod +x install_warp-ctc.sh && ./install_warp-ctc.sh ; true
	. venv/bin/activate; cd chainer_ctc && pip install .
	touch chainer_ctc.done

nkf.done:
	rm -rf nkf
	mkdir -p nkf
	cd nkf; $(WGET) https://ja.osdn.net/dl/nkf/nkf-2.1.4.tar.gz
	cd nkf; tar zxvf nkf-2.1.4.tar.gz; cd nkf-2.1.4; $(MAKE) prefix=.
	touch nkf.done

hts_engine_API.done: espnet.done
	rm -rf hts_engine_API
	git clone https://github.com/r9y9/hts_engine_API.git
	cd hts_engine_API/src && ./waf configure --prefix=../../ && ./waf build install
	touch hts_engine_API.done

open_jtalk.done: hts_engine_API.done
	rm -rf open_jtalk
	git clone https://github.com/r9y9/open_jtalk.git
	mkdir -p open_jtalk/src/build
	cd open_jtalk/src/build && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=../../../ .. && make install
	touch open_jtalk.done

pyopenjtalk.done: open_jtalk.done
	rm -rf pyopenjtalk
	git clone https://github.com/r9y9/pyopenjtalk.git
	. venv/bin/activate; cd pyopenjtalk && OPEN_JTALK_INSTALL_PREFIX=$(PWD) pip install -e .
	touch pyopenjtalk.done

moses.done:
	git clone https://github.com/moses-smt/mosesdecoder.git moses
	touch moses.done

mwerSegmenter.done:
	$(WGET) https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz
	tar zxvf mwerSegmenter.tar.gz
	rm mwerSegmenter.tar.gz
	touch mwerSegmenter.done

check_install: kaldi.done python
ifneq ($(strip $(CUPY_VERSION)),)
	. venv/bin/activate; python check_install.py
else
	. venv/bin/activate; python check_install.py --no-cupy
endif


pesq: PESQ/P862_annex_A_2005_CD/source/PESQ
PESQ/P862_annex_A_2005_CD/source/PESQ: PESQ
	cd PESQ/P862_annex_A_2005_CD/source; gcc *.c -lm -o PESQ
PESQ: PESQ.zip
	mkdir -p PESQ_P.862.2
	unzip PESQ.zip -d PESQ_P.862.2
	unzip "PESQ_P.862.2/Software/P862_annex_A_2005_CD  wav final.zip" -d PESQ_P.862.2
	rm -rf PESQ
	ln -s PESQ_P.862.2 PESQ
PESQ.zip:
	$(WGET) 'http://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-P.862-200511-I!Amd2!SOFT-ZST-E&type=items' -O PESQ.zip


clean: clean_extra
	rm -rf kaldi venv warp-ctc warp-transducer chainer_ctc
	rm -f miniconda.sh
	rm -rf *.done
	find . -iname "*.pyc" -delete

clean_python:
	rm -rf venv warp-ctc warp-transducer chainer_ctc
	rm -f miniconda.sh
	rm -f warp-ctc.done chainer_ctc.done espnet.done pytorch_complex pytorch_complex.done
	find . -iname "*.pyc" -delete

clean_extra:
	rm -rf nkf.done swig.done moses.done mwerSegmenter.done
	rm -rf hts_engine_API.done open_jtalk.done pyopenjtalk.done
	rm -rf nkf mecab swig moses mwerSegmenter
	rm -rf PESQ PESQ.zip
