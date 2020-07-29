# If a Python interpreter is specified, then creates a virtualenv from it
# PYTHON := /usr/bin/python3.7
PYTHON :=
# If you already have anaconda and you'll create an env in it,
# Give the path with "CONDA". e.g. make CONDA=/your/anaconda
CONDA := venv
CONDA_ENV_NAME := espnet
# The python version installed in the conda setup
# NOTE(kan-bayashi): Use 3.7.3 to avoid sentencepiece installation error
PYTHON_VERSION := 3.7.3
# If empty string is given, chainer is not installed. Note that ESPnet doesn't support any versions except for chainer=6.0.0
CHAINER_VERSION := 6.0.0
# Disable cupy installation
NO_CUPY :=
# PyTorch version: 0.4.1, 1.0.0, 1.0.1, 1.1.0, 1.2.0, 1.3.0, 1.3.1, 1.4.0, 1.5.0, 1.5.1, 1.6.0
TH_VERSION := 1.4.0
# Use a prebuild Kaldi to omit the installation
KALDI :=
WGET := wget --tries=3

# Use pip for pytorch installation even if you have anaconda
ifneq ($(strip $(PYTHON)),)
USE_PIP := 0
else
USE_PIP :=
endif

# Set if install binaries on CPU mode e.g. make CPU_ONLY=1
# If you don't have any GPUs, this value will be set automatically
ifeq ($(shell which nvcc),) # 'nvcc' not found
CPU_ONLY := 0
else
CPU_ONLY :=
endif

ifeq ($(strip $(CPU_ONLY)),)
# Derive CUDA version from nvcc
CUDA_VERSION = $(shell nvcc --version | grep "Cuda compilation tools" | cut -d" " -f5 | sed s/,//)
CUDA_VERSION_WITHOUT_DOT = $(strip $(subst .,,$(CUDA_VERSION)))
CONDA_PYTORCH := pytorch=$(TH_VERSION) cudatoolkit=$(CUDA_VERSION)
PIP_PYTORCH := torch==$(TH_VERSION) -f https://download.pytorch.org/whl/cu$(CUDA_VERSION_WITHOUT_DOT)/torch_stable.html

else
CONDA_PYTORCH := pytorch=$(TH_VERSION) cpuonly
PIP_PYTORCH := torch==$(TH_VERSION) -f https://download.pytorch.org/whl/cpu/torch_stable.html
NO_CUPY := 0
endif

ifeq ($(shell expr $(CUDA_VERSION_WITHOUT_DOT) \>= 102), 1)
# cupy==6.0.0 doesn't support CUDA=10.2 or later
NO_CUPY := 0
endif

ifneq ($(strip $(NO_CUPY)),)
PIP_CHAINER := chainer==$(CHAINER_VERSION)
else
PIP_CHAINER := chainer==$(CHAINER_VERSION) cupy-cuda$(CUDA_VERSION_WITHOUT_DOT)==$(CHAINER_VERSION)
endif


.PHONY: all clean

all: showenv kaldi.done python check_install

ifneq ($(strip $(CHAINER_VERSION)),)
python: activate_python.sh warp-ctc.done warp-transducer.done espnet.done pytorch.done chainer_ctc.done chainer.done
else
python: activate_python.sh warp-ctc.done warp-transducer.done espnet.done pytorch.done
endif

extra: nkf.done moses.done mwerSegmenter.done pesq kenlm.done


################ Logging ################
ifneq ($(strip $(PYTHON)),)
# venv case
showenv:
	@echo PYTHON=$(PYTHON)
else
# anaconda case
showenv:
	@echo CONDA=$(CONDA)
	@echo CONDA_ENV_NAME=$(CONDA_ENV_NAME)
	@echo PYTHON_VERSION=$(PYTHON_VERSION)
	@echo USE_PIP=$(USE_PIP)
endif
ifeq ($(strip $(CPU_ONLY)),)
	@echo CUDA_VERSION=$(CUDA_VERSION)
else
	@echo Perform on CPU mode: CPU_ONLY=$(CPU_ONLY)
endif
	@echo KALDI=$(KALDI)
	@echo TH_VERSION=$(TH_VERSION)
	@echo CONDA_PYTORCH=$(CONDA_PYTORCH)
	@echo PIP_PYTORCH=$(PIP_PYTORCH)
	@echo CHAINER_VERSION=$(CHAINER_VERSION)
	@echo PIP_CHAINER=$(PIP_CHAINER)
	@echo NO_CUPY=$(NO_CUPY)

#########################################


ifneq ($(strip $(KALDI)),)
kaldi.done:
	test -d $(KALDI)
	ln -s $(abspath $(KALDI)) kaldi  # put a symbolic link
	touch kaldi.done
else
kaldi.done:
	# make KALDI=... is mandatory now
	@echo 'Error:'
	@echo '  Please install Kaldi and'
	@echo '    # make KALDI=<kaldi-root>'
	@echo '  Where <kaldi-root> is the path of Kaldi.'
	exit 1
endif

ifneq ($(strip $(PYTHON)),)
activate_python.sh:
	test -d venv || $(PYTHON) -m venv venv
	echo ". $(CURDIR)/venv/bin/activate" > activate_python.sh
else
activate_python.sh:
	./installers/install_anaconda.sh "$(CONDA_ENV_NAME)" $(PYTHON_VERSION) $(CONDA)
	echo 'if [ -z "$${PS1:-}" ]; then' > activate_python.sh.bak
	echo '    PS1=__dummy__' >> activate_python.sh.bak
	echo 'fi' >> activate_python.sh.bak
	echo ". $(abspath $(CONDA))/etc/profile.d/conda.sh && conda deactivate && conda activate $(CONDA_ENV_NAME)" >> activate_python.sh.bak
	mv activate_python.sh.bak activate_python.sh
endif

ifneq ($(strip $(USE_PIP)),)
pytorch.done: activate_python.sh
	. ./activate_python.sh; pip install pip --upgrade
	. ./activate_python.sh; pip install numpy --upgrade  # Workaround for "ImportError: numpy.core.multiarray failed to import"
	. ./activate_python.sh; pip install $(PIP_PYTORCH)
	touch pytorch.done
else
pytorch.done: activate_python.sh
	. ./activate_python.sh && conda install -y $(CONDA_PYTORCH) -c pytorch
	touch pytorch.done
endif

espnet.done: pytorch.done
	. ./activate_python.sh && pip install -e ..  # Install editable mode by default
	touch espnet.done

chainer.done: espnet.done
	. ./activate_python.sh && pip install ${PIP_CHAINER}
	touch chainer.done

warp-ctc.done: pytorch.done
	. ./activate_python.sh && ./installers/install_warp-ctc.sh
	touch warp-ctc.done

warp-transducer.done: pytorch.done
	. ./activate_python.sh && ./installers/install_warp-transducer.sh
	touch warp-transducer.done

chainer_ctc.done: chainer.done
	. ./activate_python.sh && ./installers/install_chainer_ctc.sh
	touch chainer_ctc.done

nkf.done:
	./installers/install_nkf.sh
	touch nkf.done

pyopenjtalk.done: espnet.done
	. ./activate_python.sh && ./installers/install_pyopenjtalk.sh
	touch pyopenjtalk.done

moses.done:
	git clone https://github.com/moses-smt/mosesdecoder.git moses
	touch moses.done

mwerSegmenter.done:
	./installers/install_mwerSegmenter.sh
	touch mwerSegmenter.done

kenlm.done: espnet.done
	. ./activate_python.sh && ./installers/install_kenlm.sh
	touch kenlm.done

pesq: PESQ/P862_annex_A_2005_CD/source/PESQ
PESQ/P862_annex_A_2005_CD/source/PESQ:
	./installers/install_pesq.sh


check_install: kaldi.done python
ifeq ($(strip $(CPU_ONLY)),)

ifneq ($(strip $(NO_CUPY)),)
	. ./activate_python.sh; python check_install.py --no-cupy
else
	. ./activate_python.sh; python check_install.py
endif

else

ifneq ($(strip $(NO_CUPY)),)
	. ./activate_python.sh; python check_install.py --no-cuda --no-cupy
else
	. ./activate_python.sh; python check_install.py --no-cuda
endif

endif


clean: clean_extra
	rm -rf kaldi venv warp-ctc warp-transducer chainer_ctc
	rm -f miniconda.sh
	rm -rf *.done
	find . -iname "*.pyc" -delete

clean_python:
	rm -rf venv warp-ctc warp-transducer chainer_ctc
	rm -f miniconda.sh
	rm -f warp-ctc.done chainer_ctc.done espnet.done chainer.done pytorch.done warp-transducer.done
	find . -iname "*.pyc" -delete

clean_extra:
	rm -rf nkf.done swig.done moses.done mwerSegmenter.done
	rm -rf hts_engine_API.done open_jtalk.done pyopenjtalk.done
	rm -rf nkf mecab swig moses mwerSegmenter
	rm -rf PESQ PESQ.zip
