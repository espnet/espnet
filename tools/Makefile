# PyTorch version: The latest six versions have been tested.
# Please check https://github.com/espnet/espnet?tab=readme-ov-file#espnet-end-to-end-speech-processing-toolkit
TH_VERSION := 2.7.1

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
WITH_OMP=ON

.PHONY: all clean

all: showenv conda_packages.done python ffmpeg.done sctk check_install

python: activate_python.sh setuptools.done packaging.done espnet.done pytorch.done fairscale.done torch_optimizer.done flash_attn.done lightning.done
extra: warp-transducer.done nkf.done moses.done mwerSegmenter.done pesq kenlm.done pyopenjtalk.done py3mmseg.done beamformit.done fairseq.done s3prl.done k2.done transformers.done phonemizer.done longformer.done muskits.done whisper.done rvad_fast.done sounfile_test parallel-wavegan.done lora.done sph2pipe torcheval.done

activate_python.sh:
	@echo "::group::activate_python.sh"
	test -f activate_python.sh || { echo "Error: Run ./setup_python.sh or ./setup_miniforge.sh"; exit 1; }
	@echo "::endgroup::"

################ Logging ################
showenv: activate_python.sh
	@echo "::group::showenv"
ifeq ($(strip $(CPU_ONLY)),)
	@echo CUDA_VERSION=$(CUDA_VERSION)
else
	@echo Perform on CPU mode: CPU_ONLY=$(CPU_ONLY)
endif
	@echo PYTHON=$(shell . ./activate_python.sh && command -v python3)
	@echo PYTHON_VERSION=$(shell . ./activate_python.sh && python3 --version)
	@echo USE_CONDA=$(USE_CONDA)
	@echo TH_VERSION=$(TH_VERSION)
	@echo WITH_OMP=$(WITH_OMP)
	@echo "::endgroup::"

#########################################

bc.done: activate_python.sh
	@echo "::group::bc.done"
	. ./activate_python.sh && { command -v bc || conda install -y bc -c conda-forge; }
	touch bc.done
	@echo "::endgroup::"
cmake.done: activate_python.sh
	@echo "::group::cmake.done"
	. ./activate_python.sh && { command -v cmake || conda install -y "cmake<4.0"; }
	touch cmake.done
	@echo "::endgroup::"
flac.done: activate_python.sh
	@echo "::group::flac.done"
	. ./activate_python.sh && { command -v flac || conda install -y libflac -c conda-forge; }
	touch flac.done
	@echo "::endgroup::"
sox.done: activate_python.sh
	@echo "::group::sox.done"
	. ./activate_python.sh && { command -v sox || conda install -y sox -c conda-forge; }
	touch sox.done
	@echo "::endgroup::"
sndfile.done: activate_python.sh
	@echo "::group::sndfile.done"
	# NOTE(kamo): The wheel version of Soundfile includes shared library and it is refered in preference.
	# However, some old linuxs are not compatible with the wheel, so libsndfile is installed for this case.
	. ./activate_python.sh && { conda install -y libsndfile -c conda-forge; }
	touch sndfile.done
	@echo "::endgroup::"
ifneq ($(strip $(USE_CONDA)),)
conda_packages.done: bc.done cmake.done flac.done sox.done sndfile.done
else
conda_packages.done:
endif
	@echo "::group::conda_packages.done"
	touch conda_packages.done
	@echo "::endgroup::"

ffmpeg.done: activate_python.sh
	@echo "::group::ffmpeg.done"
ifneq ($(strip $(USE_CONDA)),)
	. ./activate_python.sh && { command -v ffmpeg || conda install -y ffmpeg -c conda-forge; }
else
	. ./activate_python.sh && { command -v ffmpeg || ./installers/install_ffmpeg.sh; }
endif
	touch ffmpeg.done
	@echo "::endgroup::"

sctk: sctk/bin/sclite
sctk/bin/sclite:
	@echo "::group::sctk/bin/sclite"
	./installers/install_sctk.sh
	@echo "::endgroup::"

sph2pipe: sph2pipe/sph2pipe
sph2pipe/sph2pipe:
	@echo "::group::sph2pipe/sph2pipe"
	./installers/install_sph2pipe.sh
	@echo "::endgroup::"

setuptools.done: activate_python.sh
	@echo "::group::setuptools.done"
	# NOTE(jiatong): For python 3.12 case, as distutils are not default from 3.12
	. ./activate_python.sh && { python -m ensurepip --upgrade; pip install setuptools; }
	touch setuptools.done
	@echo "::endgroup::"

packaging.done: setuptools.done
	@echo "::group::packaging.done"
	. ./activate_python.sh && python3 -m pip install packaging
	touch packaging.done
	@echo "::endgroup::"

numpy.done: activate_python.sh
	@echo "::group::numpy.done"
ifeq ($(strip $(USE_CONDA)),)
	. ./activate_python.sh && python3 -m pip install numpy
else
	. ./activate_python.sh && conda install -y numpy
endif
	touch numpy.done
	@echo "::endgroup::"

numba.done: numpy.done
	@echo "::group::numba.done"
	. ./activate_python.sh && python3 -m pip install -U numba
	touch numba.done
	@echo "::endgroup::"

# NOTE(kamo): Install numba before installing torch because numba requires specific numpy version now
# (Pytorch is not related to numba directly)
pytorch.done: packaging.done numba.done
	@echo "::group::pytorch.done"
ifeq ($(strip $(USE_CONDA)),)
	. ./activate_python.sh && ./installers/install_torch.sh "false" "${TH_VERSION}" "${CUDA_VERSION}"
else
	. ./activate_python.sh && ./installers/install_torch.sh "true" "${TH_VERSION}" "${CUDA_VERSION}"
endif
	touch pytorch.done
	@echo "::endgroup::"

lightning.done: pytorch.done
	@echo "::group::lightning.done"
	. ./activate_python.sh && ./installers/install_lightning.sh
	touch lightning.done
	@echo "::endgroup::"

# NOTE(kamo): conda_packages is not necessary for installation of espnet, but add it the dependencies just in case.
espnet.done: pytorch.done conda_packages.done lightning.done
	@echo "::group::espnet.done"
	@echo NUMPY_VERSION=$(shell . ./activate_python.sh && python3 -c "import numpy; print(numpy.__version__)")
	# Install additional packages that cannot be included in setup.py due to package build issues.
	. ./activate_python.sh && python3 -m pip install git+https://github.com/espnet/g2p.git@053dfa9  # https://github.com/espnet/g2p/pull/1
	. ./activate_python.sh && python3 -m pip install git+https://github.com/espnet/ctc-segmentation.git@9b9ea1d
	. ./activate_python.sh && python3 -m pip install -e ".."  # Install editable mode by default
	@echo NUMPY_VERSION=$(shell . ./activate_python.sh && python3 -c "import numpy; print(numpy.__version__)")
	. ./activate_python.sh && python -m nltk.downloader averaged_perceptron_tagger_eng
	touch espnet.done
	@echo "::endgroup::"

sounfile_test: espnet.done
	@echo "::group::sounfile_test"
	. ./activate_python.sh && python3 ./test_soundfile.py
	@echo "::endgroup::"

warp-transducer.done: pytorch.done conda_packages.done
	@echo "::group::warp-transducer.done"
ifeq ($(strip $(CPU_ONLY)),)
	[ -n "${CUDA_HOME}" ] || { echo -e "Error: CUDA_HOME is not set.\n    $$ . ./setup_cuda_env.sh <cuda-root>"; exit 1; }
endif
	. ./activate_python.sh && ./installers/install_warp-transducer.sh ${WITH_OMP}
	touch warp-transducer.done
	@echo "::endgroup::"

nkf.done:
	@echo "::group::nkf.done"
	./installers/install_nkf.sh
	touch nkf.done
	@echo "::endgroup::"

pyopenjtalk.done: espnet.done conda_packages.done
	@echo "::group::pyopenjtalk.done"
	. ./activate_python.sh && ./installers/install_pyopenjtalk.sh
	touch pyopenjtalk.done
	@echo "::endgroup::"

phonemizer.done: espnet.done conda_packages.done
	@echo "::group::phonemizer.done"
ifeq ($(WITH_OMP),ON)
	. ./activate_python.sh && ./installers/install_phonemizer.sh
	touch phonemizer.done
else
	# FIXME(kamo): I don't know how to avoid "-fopenmp" option
	echo "Warning: ./installers/install_phonemizer.sh requires WITH_OMP=ON"
endif
	@echo "::endgroup::"

speechbrain.done: espnet.done conda_packages.done
	@echo "::group::speechbrain.done"
	. ./activate_python.sh && ./installers/install_speechbrain.sh
	touch speechbrain.done
	@echo "::endgroup::"

mfa.done: espnet.done pyopenjtalk.done phonemizer.done conda_packages.done
	@echo "::group::mfa.done"
ifneq ($(strip $(USE_CONDA)),)
	. ./activate_python.sh && ./installers/install_mfa.sh "true"
else
	. ./activate_python.sh && ./installers/install_mfa.sh "false"
endif
	touch mfa.done
	@echo "::endgroup::"

moses.done:
	@echo "::group::moses.done"
	git clone --depth 1 https://github.com/moses-smt/mosesdecoder.git moses
	touch moses.done
	@echo "::endgroup::"

mwerSegmenter.done:
	@echo "::group::mwerSegmenter.done"
	./installers/install_mwerSegmenter.sh
	touch mwerSegmenter.done
	@echo "::endgroup::"

kenlm.done: espnet.done conda_packages.done
	@echo "::group::kenlm.done"
	. ./activate_python.sh && ./installers/install_kenlm.sh
	touch kenlm.done
	@echo "::endgroup::"

pesq: PESQ/P862_annex_A_2005_CD/source/PESQ
PESQ/P862_annex_A_2005_CD/source/PESQ:
	@echo "::group::PESQ/P862_annex_A_2005_CD/source/PESQ"
	./installers/install_pesq.sh
	@echo "::endgroup::"

py3mmseg.done: espnet.done
	@echo "::group::py3mmseg.done"
	. ./activate_python.sh && ./installers/install_py3mmseg.sh
	touch py3mmseg.done
	@echo "::endgroup::"

beamformit.done:
	@echo "::group::beamformit.done"
	./installers/install_beamformit.sh
	touch beamformit.done
	@echo "::endgroup::"

torch_optimizer.done: espnet.done
	@echo "::group::torch_optimizer.done"
	. ./activate_python.sh && ./installers/install_torch_optimizer.sh
	touch torch_optimizer.done
	@echo "::endgroup::"

fairscale.done: espnet.done
	@echo "::group::fairscale.done"
	. ./activate_python.sh && ./installers/install_fairscale.sh
	touch fairscale.done
	@echo "::endgroup::"

fairseq.done: espnet.done
	@echo "::group::fairseq.done"
	. ./activate_python.sh && ./installers/install_fairseq.sh
	touch fairseq.done
	@echo "::endgroup::"

s3prl.done: espnet.done
	@echo "::group::s3prl.done"
	. ./activate_python.sh && ./installers/install_s3prl.sh
	touch s3prl.done
	@echo "::endgroup::"

whisper.done: espnet.done
	@echo "::group::whisper.done"
	. ./activate_python.sh && ./installers/install_whisper.sh
	touch whisper.done
	@echo "::endgroup::"

lora.done: espnet.done
	@echo "::group::lora.done"
	. ./activate_python.sh && ./installers/install_lora.sh
	touch lora.done
	@echo "::endgroup::"

k2.done: espnet.done
	@echo "::group::k2.done"
	. ./activate_python.sh && ./installers/install_k2.sh
	touch k2.done
	@echo "::endgroup::"

gtn.done: espnet.done
	@echo "::group::gtn.done"
	. ./activate_python.sh && ./installers/install_gtn.sh
	touch gtn.done
	@echo "::endgroup::"

transformers.done: espnet.done
	@echo "::group::transformers.done"
	. ./activate_python.sh && ./installers/install_transformers.sh
	touch transformers.done
	@echo "::endgroup::"

torcheval.done: pytorch.done
	@echo "::group::torcheval.done"
	. ./activate_python.sh && ./installers/install_torcheval.sh
	touch torcheval.done
	@echo "::endgroup::"

parallel-wavegan.done: espnet.done
	@echo "::group::parallel-wavegan.done"
	. ./activate_python.sh && ./installers/install_parallel-wavegan.sh
	touch parallel-wavegan.done
	@echo "::endgroup::"

longformer.done: espnet.done
	@echo "::group::longformer.done"
	. ./activate_python.sh && ./installers/install_longformer.sh
	touch longformer.done
	@echo "::endgroup::"

flash_attn.done: espnet.done
	@echo "::group::flash_attn.done"
	. ./activate_python.sh && ./installers/install_flash_attn.sh
	touch flash_attn.done
	@echo "::endgroup::"

muskits.done: espnet.done
	@echo "::group::muskits.done"
	. ./activate_python.sh && ./installers/install_muskits.sh
	touch muskits.done
	@echo "::endgroup::"

cauchy_mult.done: espnet.done
	@echo "::group::cauchy_mult.done"
	. ./activate_python.sh && ./installers/install_cauchy_mult.sh
	touch cauchy_mult.done
	@echo "::endgroup::"

rvad_fast.done: espnet.done
	@echo "::group::rvad_fast.done"
	git clone https://github.com/zhenghuatan/rVADfast.git
	touch rvad_fast.done
	@echo "::endgroup::"

discrete_speech_metrics.done: espnet.done
	@echo "::group::discrete_speech_metrics.done"
	. ./activate_python.sh && ./installers/install_discrete_speech_metrics.sh
	touch discrete_speech_metrics.done
	@echo "::endgroup::"

versa.done: espnet.done
	@echo "::group::versa.done"
	. ./activate_python.sh && ./installers/install_versa.sh
	touch versa.done
	@echo "::endgroup::"

check_install: python
	@echo "::group::check_install"
	. ./activate_python.sh; . ./extra_path.sh; python3 check_install.py
	@echo "::endgroup::"

clean: clean_extra
	rm -rf warp-transducer
	rm -rf *.done
	find . -iname "*.pyc" -delete

clean_python:
	rm -rf warp-transducer
	rm -f espnet.done pytorch.done warp-transducer.done
	find . -iname "*.pyc" -delete

clean_extra:
	rm -rf nkf.done swig.done moses.done mwerSegmenter.done
	rm -rf hts_engine_API.done open_jtalk.done pyopenjtalk.done
	rm -rf muskits.done
	rm -rf rvad_fast.done
	rm -rf lightning_constraints.txt
	rm -rf espeak-ng festival MBROLA ParallelWaveGAN versa
	rm -rf py3mmseg sctk* speech_tools sph2pipe* ._mwerSegmenter
	rm -rf nkf mecab swig moses mwerSegmenter
	rm -rf PESQ PESQ.zip
