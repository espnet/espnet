# If a Python interpreter is specified, then creates a virtualenv from it
# PYTHON := /usr/bin/python3.7
PYTHON :=
# The python version installed in the conda setup
# NOTE(kan-bayashi): Use 3.7.3 to avoid sentencepiece installation error
PYTHON_VERSION := 3.7.3
CUPY_VERSION := 6.0.0
CUDA_VERSION := 10.0
# PyTorch version: 0.4.1, 1.0.0, 1.0.1, 1.1.0, 1.2.0, 1.3.0, 1.3.1, 1.4.0
TH_VERSION := 1.0.1
# Use a prebuild Kaldi to omit the installation
KALDI :=
WGET := wget --tries=3


ifneq ($(shell which nvidia-smi),) # 'nvcc' found
CONDA_PYTORCH := pytorch=$(TH_VERSION) cudatoolkit=$(CUDA_VERSION)
else
CONDA_PYTORCH := pytorch=$(TH_VERSION) cpuonly
CUPY_VERSION :=
endif


.PHONY: all clean

all: kaldi.done python check_install

python: venv warp-ctc.done warp-transducer.done chainer_ctc.done

extra: nkf.done moses.done mwerSegmenter.done pesq kenlm.done

ifneq ($(strip $(KALDI)),)
kaldi.done:
	test -d $(KALDI)
	ln -s $(abspath $(KALDI)) kaldi
	touch kaldi.done
else
kaldi.done:
	test -d kaldi || git --depth 1 clone https://github.com/kaldi-asr/kaldi.git
	cd kaldi/tools; $(MAKE) all
	cd kaldi/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
	touch kaldi.done
endif


ifneq ($(strip $(PYTHON)),)
venv:
	test -d venv || $(PYTHON) -m venv venv
espnet.done: venv
	. venv/bin/activate; pip install pip --upgrade
	. venv/bin/activate; pip install torch==$(TH_VERSION)
	. venv/bin/activate; pip install -e ..
	touch espnet.done
else
venv:
	./install_anaconda.sh $(PYTHON_VERSION) venv
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
	. venv/bin/activate && ./install_warp-ctc.sh
	touch warp-ctc.done

warp-transducer.done: espnet.done
	. venv/bin/activate && ./install_warp-transducer.sh
	touch warp-transducer.done

chainer_ctc.done: espnet.done
	. venv/bin/activate && ./install_chainer_ctc.sh
	touch chainer_ctc.done

nkf.done:
	./install_nkf.sh
	touch nkf.done

pyopenjtalk.done: open_jtalk.done
	. venv/bin/activate && ./install_pyopenjtalk.sh
	touch pyopenjtalk.done

moses.done:
	git clone https://github.com/moses-smt/mosesdecoder.git moses
	touch moses.done

mwerSegmenter.done:
	./install_mwerSegmenter.sh
	touch mwerSegmenter.done

kenlm.done:
	. venv/bin/activate && ./install_kenlm.sh
	touch kenlm.done

pesq: PESQ/P862_annex_A_2005_CD/source/PESQ
PESQ/P862_annex_A_2005_CD/source/PESQ:
	./install_pesq.sh


check_install: kaldi.done python
ifneq ($(strip $(CUPY_VERSION)),)
	. venv/bin/activate; python check_install.py
else
	. venv/bin/activate; python check_install.py --no-cuda
endif


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
