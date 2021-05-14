# If a Python interpreter is specified, then creates a virtualenv from it
# If empty string is given, chainer is not installed. Note that ESPnet doesn't support any versions except for chainer=6.0.0
CHAINER_VERSION := 6.0.0
# Disable cupy installation
NO_CUPY :=
# PyTorch version: 0.4.1, 1.0.0, 1.0.1, 1.1.0, 1.2.0, 1.3.0, 1.3.1, 1.4.0, 1.5.0, 1.5.1, 1.6.0, 1.7.0, 1.7.1, 1.8.0, 1.8.1
TH_VERSION := 1.4.0
WGET := wget --tries=3

# Use pip for pytorch installation even if you have anaconda
ifneq ($(shell test -f ./activate_python.sh && grep 'conda activate' ./activate_python.sh),)
IS_CONDA := 0
USE_PIP :=
else
IS_CONDA :=
USE_PIP := 0
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
PIP_PYTORCH := torch==$(TH_VERSION) -f https://download.pytorch.org/whl/cu$(CUDA_VERSION_WITHOUT_DOT)/torch_stable.html
ifeq ($(strip $(CUDA_VERSION_WITHOUT_DOT)),80)
CONDA_PYTORCH := pytorch=$(TH_VERSION) cuda$(CUDA_VERSION_WITHOUT_DOT) -c pytorch
else ifeq ($(strip $(CUDA_VERSION_WITHOUT_DOT)),111)
CONDA_PYTORCH := pytorch=$(TH_VERSION) cudatoolkit=$(CUDA_VERSION) -c pytorch -c conda-forge
else
CONDA_PYTORCH := pytorch=$(TH_VERSION) cudatoolkit=$(CUDA_VERSION) -c pytorch
endif

else
CONDA_PYTORCH := pytorch=$(TH_VERSION) cpuonly -c pytorch
PIP_PYTORCH := torch==$(TH_VERSION) -f https://download.pytorch.org/whl/cpu/torch_stable.html
NO_CUPY := 0
endif

ifeq ($(shell expr $(CUDA_VERSION_WITHOUT_DOT) \>= 102), 1)
# cupy==6.0.0 doesn't support CUDA=10.2 or later
NO_CUPY := 0
endif
PIP_CHAINER := chainer==$(CHAINER_VERSION)


.PHONY: all clean

all: kaldi showenv python conda_packages.done sctk.done sph2pipe.done check_install

ifneq ($(strip $(CHAINER_VERSION)),)
python: activate_python.sh espnet.done pytorch.done chainer.done
extra: warp-ctc.done warp-transducer.done chainer_ctc.done nkf.done moses.done mwerSegmenter.done pesq kenlm.done pyopenjtalk.done py3mmseg.done beamformit.done fairseq.done
else
python: activate_python.sh espnet.done pytorch.done
extra: warp-ctc.done warp-transducer.done nkf.done moses.done mwerSegmenter.done pesq kenlm.done pyopenjtalk.done py3mmseg.done beamformit.done fairseq.done
endif


kaldi:
	test -f kaldi/egs/wsj/s5/utils/parse_options.sh || git clone --depth 1 https://github.com/kaldi-asr/kaldi

activate_python.sh:
	test -f activate_python.sh || { echo "Error: Run ./setup_python.sh or ./setup_anaconda.sh"; exit 1; }

################ Logging ################
showenv: activate_python.sh
ifeq ($(strip $(CPU_ONLY)),)
	@echo CUDA_VERSION=$(CUDA_VERSION)
	. ./activate_python.sh && python3 check_pytorch_cuda_compatibility.py $(TH_VERSION) ${CUDA_VERSION} || { echo "Change TH_VERSION or CUDA_VERSION"; exit 1; }
else
	@echo Perform on CPU mode: CPU_ONLY=$(CPU_ONLY)
endif
	@echo PYTHON=$(shell . ./activate_python.sh && command -v python3)
	@echo PYTHON_VERSION=$(shell . ./activate_python.sh && python3 --version)
	@echo IS_CONDA=$(IS_CONDA)
	@echo USE_PIP=$(USE_PIP)
	@echo TH_VERSION=$(TH_VERSION)
	@echo CONDA_PYTORCH=$(CONDA_PYTORCH)
	@echo PIP_PYTORCH=$(PIP_PYTORCH)
	@echo CHAINER_VERSION=$(CHAINER_VERSION)
	@echo PIP_CHAINER=$(PIP_CHAINER)
	@echo NO_CUPY=$(NO_CUPY)

#########################################

cmake.done: activate_python.sh
	. ./activate_python.sh && { command -v cmake || conda install -y cmake; }
	touch cmake.done
flac.done: activate_python.sh
	. ./activate_python.sh && { command -v flac || conda install -y libflac -c conda-forge; }
	touch flac.done
ffmpeg.done: activate_python.sh
	. ./activate_python.sh && { command -v ffmpeg || conda install -y ffmpeg -c conda-forge; }
	touch ffmpeg.done
sox.done: activate_python.sh
	. ./activate_python.sh && { command -v sox || conda install -y sox -c conda-forge; }
	touch sox.done
sndfile.done: activate_python.sh
	. ./activate_python.sh && { python3 -c "from ctypes.util import find_library as F; assert F('sndfile') is not None" || conda install -y libsndfile=1.0.28 -c conda-forge; }
	touch sndfile.done
ifneq ($(strip $(IS_CONDA)),)
conda_packages.done: cmake.done flac.done ffmpeg.done sox.done sndfile.done
else
conda_packages.done:
endif
	touch conda_packages.done


sctk.done:
	./installers/install_sctk.sh
	touch sctk.done
sph2pipe.done:
	./installers/install_sph2pipe.sh
	touch sph2pipe.done


ifneq ($(strip $(USE_PIP)),)
pytorch.done: activate_python.sh
	. ./activate_python.sh && python3 -m pip install numpy --upgrade  # Workaround for "ImportError: numpy.core.multiarray failed to import"
	. ./activate_python.sh && python3 -m pip install $(PIP_PYTORCH)
	touch pytorch.done
else
pytorch.done: activate_python.sh
	. ./activate_python.sh && conda install -y $(CONDA_PYTORCH)
	touch pytorch.done
endif

# NOTE(kamo): conda_packages is not necessary for installation of espnet, but add it the dependencies just in case.
espnet.done: pytorch.done conda_packages.done
	. ./activate_python.sh && python3 -m pip install -e "..[recipe]"  # Install editable mode by default
	touch espnet.done

chainer.done: espnet.done
ifneq ($(strip $(NO_CUPY)),)
	. ./activate_python.sh && python3 -m pip install $(PIP_CHAINER)
else
	# Precompiled cupy==6.0.0 for python>=3.8 is not provided
	. ./activate_python.sh && \
		if python3 -c "import sys; from distutils.version import LooseVersion as L; assert L(sys.version) < L('3.8')" 2>&1 /dev/null; then \
			python3 -m pip install $(PIP_CHAINER) cupy-cuda$(CUDA_VERSION_WITHOUT_DOT)==$(CHAINER_VERSION); \
		else \
			python3 -m pip install $(PIP_CHAINER) cupy==$(CHAINER_VERSION); \
		fi
endif
	# chainer=6.0.0 depends on typing<=3.6.6, but this causes the following error when installing some modules. e.g. fairseq
	# AttributeError: type object 'Callable' has no attribute '_abc_registry'
	# "typing" modules is not required for python>=3.6, so uninstall here
	. ./activate_python.sh && python3 -m pip uninstall -y typing
	touch chainer.done

# NOTE(kamo): Add conda_packages.done if cmake is used
warp-ctc.done: pytorch.done conda_packages.done
ifeq ($(strip $(CPU_ONLY)),)
	[ -n "${CUDA_HOME}" ] || { echo -e "Error: CUDA_HOME is not set.\n    $$ . ./setup_cuda_env.sh <cuda-root>"; exit 1; }
endif
	. ./activate_python.sh && ./installers/install_warp-ctc.sh
	touch warp-ctc.done

warp-transducer.done: pytorch.done conda_packages.done
ifeq ($(strip $(CPU_ONLY)),)
	[ -n "${CUDA_HOME}" ] || { echo -e "Error: CUDA_HOME is not set.\n    $$ . ./setup_cuda_env.sh <cuda-root>"; exit 1; }
	. ./activate_python.sh && ./installers/install_warp-transducer.sh
	. ./activate_python.sh &&  ./installers/install_warp-rnnt.sh $(CUDA_VERSION)
	touch warp-transducer.done
else
	. ./activate_python.sh && ./installers/install_warp-transducer.sh
	touch warp-transducer.done
endif

chainer_ctc.done: chainer.done
ifeq ($(strip $(CPU_ONLY)),)
	[ -n "${CUDA_HOME}" ] || { echo -e "Error: CUDA_HOME is not set.\n    $$ . ./setup_cuda_env.sh <cuda-root>"; exit 1; }
endif
	. ./activate_python.sh && ./installers/install_chainer_ctc.sh
	touch chainer_ctc.done

nkf.done:
	./installers/install_nkf.sh
	touch nkf.done

pyopenjtalk.done: espnet.done conda_packages.done
	. ./activate_python.sh && ./installers/install_pyopenjtalk.sh
	touch pyopenjtalk.done

moses.done:
	git clone --depth 1 https://github.com/moses-smt/mosesdecoder.git moses
	touch moses.done

mwerSegmenter.done:
	./installers/install_mwerSegmenter.sh
	touch mwerSegmenter.done

kenlm.done: espnet.done conda_packages.done
	. ./activate_python.sh && ./installers/install_kenlm.sh
	touch kenlm.done

pesq: PESQ/P862_annex_A_2005_CD/source/PESQ
PESQ/P862_annex_A_2005_CD/source/PESQ:
	./installers/install_pesq.sh

py3mmseg.done: espnet.done
	. ./activate_python.sh && ./installers/install_py3mmseg.sh
	touch py3mmseg.done

beamformit.done:
	./installers/install_beamformit.sh
	touch beamformit.done

fairseq.done: espnet.done
	. ./activate_python.sh && ./installers/install_fairseq.sh
	touch fairseq.done

check_install: python
	. ./activate_python.sh; . ./extra_path.sh; python3 check_install.py


clean: clean_extra
	rm -rf warp-ctc warp-transducer warp-rnnt chainer_ctc
	rm -rf *.done
	find . -iname "*.pyc" -delete

clean_python:
	rm -rf warp-ctc warp-transducer warp-rnnt chainer_ctc
	rm -f warp-ctc.done chainer_ctc.done espnet.done chainer.done pytorch.done warp-transducer.done
	find . -iname "*.pyc" -delete

clean_extra:
	rm -rf nkf.done swig.done moses.done mwerSegmenter.done
	rm -rf hts_engine_API.done open_jtalk.done pyopenjtalk.done
	rm -rf nkf mecab swig moses mwerSegmenter
	rm -rf PESQ PESQ.zip
