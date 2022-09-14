# PyTorch version: 1.4.0, 1.5.1, 1.6.0, 1.7.1, 1.8.1, 1.9.1, 1.10.0, 1.10.1, 1.11.0 and 1.12.1 are tested.
TH_VERSION := 1.12.1

# Use pip for pytorch installation even if you have anaconda
ifneq ($(shell test -f ./activate_python.sh && grep 'conda activate' ./activate_python.sh),)
USE_CONDA := 1
else
USE_CONDA :=
endif


# Set if install binaries on CPU mode e.g. make CPU_ONLY=0
# If you don't have nvcc, this value will be set automatically
ifneq ($(shell which nvcc 2>/dev/null),)
CPU_ONLY :=
# Derive CUDA version from nvcc
CUDA_VERSION := $(shell nvcc --version | grep "Cuda compilation tools" | cut -d" " -f5 | sed s/,//)
CUDA_VERSION_WITHOUT_DOT := $(strip $(subst .,,$(CUDA_VERSION)))

else
CPU_ONLY := 0
CUDA_VERSION :=
CUDA_VERSION_WITHOUT_DOT :=
endif


.PHONY: all clean

all: kaldi showenv python conda_packages.done sctk.done sph2pipe.done check_install

python: activate_python.sh packaging.done espnet.done pytorch.done chainer.done fairscale.done torch_optimizer.done
extra: warp-ctc.done warp-transducer.done chainer_ctc.done nkf.done moses.done mwerSegmenter.done pesq kenlm.done pyopenjtalk.done py3mmseg.done beamformit.done fairseq.done s3prl.done k2.done transformers.done phonemizer.done longformer.done

kaldi:
	test -f kaldi/egs/wsj/s5/utils/parse_options.sh || git clone --depth 1 https://github.com/kaldi-asr/kaldi

activate_python.sh:
	test -f activate_python.sh || { echo "Error: Run ./setup_python.sh or ./setup_anaconda.sh"; exit 1; }

################ Logging ################
showenv: activate_python.sh
ifeq ($(strip $(CPU_ONLY)),)
	@echo CUDA_VERSION=$(CUDA_VERSION)
else
	@echo Perform on CPU mode: CPU_ONLY=$(CPU_ONLY)
endif
	@echo PYTHON=$(shell . ./activate_python.sh && command -v python3)
	@echo PYTHON_VERSION=$(shell . ./activate_python.sh && python3 --version)
	@echo USE_CONDA=$(USE_CONDA)
	@echo TH_VERSION=$(TH_VERSION)

#########################################

bc.done: activate_python.sh
	. ./activate_python.sh && { command -v bc || conda install -y bc -c conda-forge; }
	touch bc.done
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
	. ./activate_python.sh && { python3 -c "from ctypes.util import find_library as F; assert F('sndfile') is not None" || conda install -y libsndfile -c conda-forge; }
	touch sndfile.done
ifneq ($(strip $(USE_CONDA)),)
conda_packages.done: bc.done cmake.done flac.done ffmpeg.done sox.done sndfile.done
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

packaging.done: activate_python.sh
	. ./activate_python.sh && python3 -m pip install packaging
	touch packaging.done

numba.done: activate_python.sh
	. ./activate_python.sh && python3 -m pip install -U numba
	touch numba.done

# NOTE(kamo): Install numba before installing torch because numba requires specific numpy version now
# (Pytorch is not related to numba directly)
pytorch.done: packaging.done numba.done
ifeq ($(strip $(USE_CONDA)),)
	. ./activate_python.sh && ./installers/install_torch.sh "false" "${TH_VERSION}" "${CUDA_VERSION}"
else
	. ./activate_python.sh && ./installers/install_torch.sh "true" "${TH_VERSION}" "${CUDA_VERSION}"
endif
	touch pytorch.done

# NOTE(kamo): conda_packages is not necessary for installation of espnet, but add it the dependencies just in case.
espnet.done: pytorch.done conda_packages.done
	@echo NUMPY_VERSION=$(shell . ./activate_python.sh && python3 -c "import numpy; print(numpy.__version__)")
	. ./activate_python.sh && python3 -m pip install -e "..[train, recipe]"  # Install editable mode by default
	@echo NUMPY_VERSION=$(shell . ./activate_python.sh && python3 -c "import numpy; print(numpy.__version__)")
	touch espnet.done

chainer.done: espnet.done
	. ./activate_python.sh && ./installers/install_chainer.sh "${CUDA_VERSION}"
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

phonemizer.done: espnet.done conda_packages.done
	. ./activate_python.sh && ./installers/install_phonemizer.sh
	touch phonemizer.done

speechbrain.done: espnet.done conda_packages.done
	. ./activate_python.sh && ./installers/install_speechbrain.sh
	touch speechbrain.done

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

torch_optimizer.done: espnet.done
	. ./activate_python.sh && ./installers/install_torch_optimizer.sh
	touch torch_optimizer.done

fairscale.done: espnet.done
	. ./activate_python.sh && ./installers/install_fairscale.sh
	touch fairscale.done

fairseq.done: espnet.done
	. ./activate_python.sh && ./installers/install_fairseq.sh
	touch fairseq.done

s3prl.done: espnet.done
	. ./activate_python.sh && ./installers/install_s3prl.sh
	touch s3prl.done

k2.done: espnet.done
	. ./activate_python.sh && ./installers/install_k2.sh
	touch k2.done

gtn.done: espnet.done
	. ./activate_python.sh && ./installers/install_gtn.sh
	touch gtn.done

transformers.done: espnet.done
	. ./activate_python.sh && ./installers/install_transformers.sh
	touch transformers.done

longformer.done: espnet.done
	. ./activate_python.sh && ./installers/install_longformer.sh
	touch longformer.done

check_install: python
	. ./activate_python.sh; . ./extra_path.sh; python3 check_install.py

clean: clean_extra
	rm -rf warp-ctc warp-transducer chainer_ctc
	rm -rf *.done
	find . -iname "*.pyc" -delete

clean_python:
	rm -rf warp-ctc warp-transducer chainer_ctc
	rm -f warp-ctc.done chainer_ctc.done espnet.done chainer.done pytorch.done warp-transducer.done
	find . -iname "*.pyc" -delete

clean_extra:
	rm -rf nkf.done swig.done moses.done mwerSegmenter.done
	rm -rf hts_engine_API.done open_jtalk.done pyopenjtalk.done
	rm -rf nkf mecab swig moses mwerSegmenter
	rm -rf PESQ PESQ.zip
