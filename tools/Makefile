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

# Timer functions for measuring task execution time
define start_timer
	@echo "::group::$(1)"
	@date +%s > .$(subst /,_,$(1))_start_time
endef

define end_timer_and_touch
	@echo "::endgroup::"
	@END_TIME=$$(date +%s); START_TIME=$$(cat .$(subst /,_,$(1))_start_time); ELAPSED=$$(($$END_TIME - $$START_TIME)); printf "⏱️  $(1) completed in %02d:%02d:%02d\n" $$(($$ELAPSED/3600)) $$(($$ELAPSED%3600/60)) $$(($$ELAPSED%60)); rm -f .$(subst /,_,$(1))_start_time
	@touch $(subst /,_,$(1))
endef

.PHONY: all clean

all: showenv conda_packages.done python ffmpeg.done sctk check_install

python: activate_python.sh setuptools.done packaging.done espnet.done pytorch.done fairscale.done torch_optimizer.done flash_attn.done lightning.done
extra: warp-transducer.done nkf.done moses.done mwerSegmenter.done pesq kenlm.done pyopenjtalk.done py3mmseg.done beamformit.done fairseq.done s3prl.done k2.done transformers.done phonemizer.done longformer.done muskits.done whisper.done rvad_fast.done sounfile_test parallel-wavegan.done lora.done sph2pipe torcheval.done

activate_python.sh:
	test -f activate_python.sh || { echo "Error: Run ./setup_python.sh or ./setup_miniforge.sh"; exit 1; }

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
	@echo WITH_OMP=$(WITH_OMP)

#########################################

bc.done: activate_python.sh
	$(call start_timer,bc.done)
	. ./activate_python.sh && { command -v bc || conda install -y bc -c conda-forge; }
	$(call end_timer_and_touch,bc.done)
cmake.done: activate_python.sh
	$(call start_timer,cmake.done)
	. ./activate_python.sh && { command -v cmake || conda install -y "cmake<4.0"; }
	$(call end_timer_and_touch,cmake.done)
flac.done: activate_python.sh
	$(call start_timer,flac.done)
	. ./activate_python.sh && { command -v flac || conda install -y libflac -c conda-forge; }
	$(call end_timer_and_touch,flac.done)
sox.done: activate_python.sh
	$(call start_timer,sox.done)
	. ./activate_python.sh && { command -v sox || conda install -y sox -c conda-forge; }
	$(call end_timer_and_touch,sox.done)
sndfile.done: activate_python.sh
	$(call start_timer,sndfile.done)
	# NOTE(kamo): The wheel version of Soundfile includes shared library and it is refered in preference.
	# However, some old linuxs are not compatible with the wheel, so libsndfile is installed for this case.
	. ./activate_python.sh && { conda install -y libsndfile -c conda-forge; }
	$(call end_timer_and_touch,sndfile.done)
ifneq ($(strip $(USE_CONDA)),)
conda_packages.done: bc.done cmake.done flac.done sox.done sndfile.done
else
conda_packages.done:
endif
	$(call start_timer,conda_packages.done)
	$(call end_timer_and_touch,conda_packages.done)

ffmpeg.done: activate_python.sh
	$(call start_timer,ffmpeg.done)
ifneq ($(strip $(USE_CONDA)),)
	. ./activate_python.sh && { command -v ffmpeg || conda install -y ffmpeg -c conda-forge; }
else
	. ./activate_python.sh && { command -v ffmpeg || ./installers/install_ffmpeg.sh; }
endif
	$(call end_timer_and_touch,ffmpeg.done)

sctk: sctk/bin/sclite
sctk/bin/sclite:
	$(call start_timer,sctk/bin/sclite)
	./installers/install_sctk.sh
	$(call end_timer_and_touch,sctk/bin/sclite)

sph2pipe: sph2pipe/sph2pipe
sph2pipe/sph2pipe:
	$(call start_timer,sph2pipe/sph2pipe)
	./installers/install_sph2pipe.sh
	$(call end_timer_and_touch,sph2pipe/sph2pipe)

setuptools.done: activate_python.sh
	$(call start_timer,setuptools.done)
	# NOTE(jiatong): For python 3.12 case, as distutils are not default from 3.12
	. ./activate_python.sh && { python -m ensurepip --upgrade; pip install setuptools; }
	$(call end_timer_and_touch,setuptools.done)

packaging.done: setuptools.done
	$(call start_timer,packaging.done)
	. ./activate_python.sh && python3 -m pip install packaging
	$(call end_timer_and_touch,packaging.done)

numpy.done: activate_python.sh
	$(call start_timer,numpy.done)
ifeq ($(strip $(USE_CONDA)),)
	. ./activate_python.sh && python3 -m pip install numpy
else
	. ./activate_python.sh && conda install -y numpy
endif
	$(call end_timer_and_touch,numpy.done)

numba.done: numpy.done
	$(call start_timer,numba.done)
	. ./activate_python.sh && python3 -m pip install -U numba
	$(call end_timer_and_touch,numba.done)

# NOTE(kamo): Install numba before installing torch because numba requires specific numpy version now
# (Pytorch is not related to numba directly)
pytorch.done: packaging.done numba.done
	$(call start_timer,pytorch.done)
ifeq ($(strip $(USE_CONDA)),)
	. ./activate_python.sh && ./installers/install_torch.sh "false" "${TH_VERSION}" "${CUDA_VERSION}"
else
	. ./activate_python.sh && ./installers/install_torch.sh "true" "${TH_VERSION}" "${CUDA_VERSION}"
endif
	$(call end_timer_and_touch,pytorch.done)

lightning.done: pytorch.done
	$(call start_timer,lightning.done)
	. ./activate_python.sh && ./installers/install_lightning.sh
	$(call end_timer_and_touch,lightning.done)

# NOTE(kamo): conda_packages is not necessary for installation of espnet, but add it the dependencies just in case.
espnet.done: pytorch.done conda_packages.done lightning.done
	$(call start_timer,espnet.done)
	@echo NUMPY_VERSION=$(shell . ./activate_python.sh && python3 -c "import numpy; print(numpy.__version__)")
	# Install additional packages that cannot be included in setup.py due to package build issues.
	. ./activate_python.sh && python3 -m pip install git+https://github.com/espnet/g2p.git@053dfa9  # https://github.com/espnet/g2p/pull/1
	. ./activate_python.sh && python3 -m pip install git+https://github.com/espnet/ctc-segmentation.git@9b9ea1d
	. ./activate_python.sh && python3 -m pip install -e ".."  # Install editable mode by default
	@echo NUMPY_VERSION=$(shell . ./activate_python.sh && python3 -c "import numpy; print(numpy.__version__)")
	. ./activate_python.sh && python -m nltk.downloader averaged_perceptron_tagger_eng
	$(call end_timer_and_touch,espnet.done)

sounfile_test: espnet.done
	$(call start_timer,sounfile_test)
	. ./activate_python.sh && python3 ./test_soundfile.py
	$(call end_timer_and_touch,sounfile_test)

warp-transducer.done: pytorch.done conda_packages.done
	$(call start_timer,warp-transducer.done)
ifeq ($(strip $(CPU_ONLY)),)
	[ -n "${CUDA_HOME}" ] || { echo -e "Error: CUDA_HOME is not set.\n    $$ . ./setup_cuda_env.sh <cuda-root>"; exit 1; }
endif
	. ./activate_python.sh && ./installers/install_warp-transducer.sh ${WITH_OMP}
	$(call end_timer_and_touch,warp-transducer.done)

nkf.done:
	$(call start_timer,nkf.done)
	./installers/install_nkf.sh
	$(call end_timer_and_touch,nkf.done)

pyopenjtalk.done: espnet.done conda_packages.done
	$(call start_timer,pyopenjtalk.done)
	. ./activate_python.sh && ./installers/install_pyopenjtalk.sh
	$(call end_timer_and_touch,pyopenjtalk.done)

phonemizer.done: espnet.done conda_packages.done
	$(call start_timer,phonemizer.done)
ifeq ($(WITH_OMP),ON)
	. ./activate_python.sh && ./installers/install_phonemizer.sh
else
	# FIXME(kamo): I don't know how to avoid "-fopenmp" option
	echo "Warning: ./installers/install_phonemizer.sh requires WITH_OMP=ON"
endif
	$(call end_timer_and_touch,phonemizer.done)

speechbrain.done: espnet.done conda_packages.done
	$(call start_timer,speechbrain.done)
	. ./activate_python.sh && ./installers/install_speechbrain.sh
	$(call end_timer_and_touch,speechbrain.done)

mfa.done: espnet.done pyopenjtalk.done phonemizer.done conda_packages.done
	$(call start_timer,mfa.done)
ifneq ($(strip $(USE_CONDA)),)
	. ./activate_python.sh && ./installers/install_mfa.sh "true"
else
	. ./activate_python.sh && ./installers/install_mfa.sh "false"
endif
	$(call end_timer_and_touch,mfa.done)

moses.done:
	$(call start_timer,moses.done)
	git clone --depth 1 https://github.com/moses-smt/mosesdecoder.git moses
	$(call end_timer_and_touch,moses.done)

mwerSegmenter.done:
	$(call start_timer,mwerSegmenter.done)
	./installers/install_mwerSegmenter.sh
	$(call end_timer_and_touch,mwerSegmenter.done)

kenlm.done: espnet.done conda_packages.done
	$(call start_timer,kenlm.done)
	. ./activate_python.sh && ./installers/install_kenlm.sh
	$(call end_timer_and_touch,kenlm.done)

pesq: PESQ/P862_annex_A_2005_CD/source/PESQ
PESQ/P862_annex_A_2005_CD/source/PESQ:
	$(call start_timer,PESQ/P862_annex_A_2005_CD/source/PESQ)
	./installers/install_pesq.sh
	$(call end_timer_and_touch,PESQ/P862_annex_A_2005_CD/source/PESQ)

py3mmseg.done: espnet.done
	$(call start_timer,py3mmseg.done)
	. ./activate_python.sh && ./installers/install_py3mmseg.sh
	$(call end_timer_and_touch,py3mmseg.done)

beamformit.done:
	$(call start_timer,beamformit.done)
	./installers/install_beamformit.sh
	$(call end_timer_and_touch,beamformit.done)

torch_optimizer.done: espnet.done
	$(call start_timer,torch_optimizer.done)
	. ./activate_python.sh && ./installers/install_torch_optimizer.sh
	$(call end_timer_and_touch,torch_optimizer.done)

fairscale.done: espnet.done
	$(call start_timer,fairscale.done)
	. ./activate_python.sh && ./installers/install_fairscale.sh
	$(call end_timer_and_touch,fairscale.done)

fairseq.done: espnet.done
	$(call start_timer,fairseq.done)
	. ./activate_python.sh && ./installers/install_fairseq.sh
	$(call end_timer_and_touch,fairseq.done)

s3prl.done: espnet.done
	$(call start_timer,s3prl.done)
	. ./activate_python.sh && ./installers/install_s3prl.sh
	$(call end_timer_and_touch,s3prl.done)

whisper.done: espnet.done
	$(call start_timer,whisper.done)
	. ./activate_python.sh && ./installers/install_whisper.sh
	$(call end_timer_and_touch,whisper.done)

lora.done: espnet.done
	$(call start_timer,lora.done)
	. ./activate_python.sh && ./installers/install_lora.sh
	$(call end_timer_and_touch,lora.done)

k2.done: espnet.done
	$(call start_timer,k2.done)
	. ./activate_python.sh && ./installers/install_k2.sh
	$(call end_timer_and_touch,k2.done)

gtn.done: espnet.done
	$(call start_timer,gtn.done)
	. ./activate_python.sh && ./installers/install_gtn.sh
	$(call end_timer_and_touch,gtn.done)

transformers.done: espnet.done
	$(call start_timer,transformers.done)
	. ./activate_python.sh && ./installers/install_transformers.sh
	$(call end_timer_and_touch,transformers.done)

torcheval.done: pytorch.done
	$(call start_timer,torcheval.done)
	. ./activate_python.sh && ./installers/install_torcheval.sh
	$(call end_timer_and_touch,torcheval.done)

parallel-wavegan.done: espnet.done
	$(call start_timer,parallel-wavegan.done)
	. ./activate_python.sh && ./installers/install_parallel-wavegan.sh
	$(call end_timer_and_touch,parallel-wavegan.done)

longformer.done: espnet.done
	$(call start_timer,longformer.done)
	. ./activate_python.sh && ./installers/install_longformer.sh
	$(call end_timer_and_touch,longformer.done)

flash_attn.done: espnet.done
	$(call start_timer,flash_attn.done)
	. ./activate_python.sh && ./installers/install_flash_attn.sh
	$(call end_timer_and_touch,flash_attn.done)

muskits.done: espnet.done
	$(call start_timer,muskits.done)
	. ./activate_python.sh && ./installers/install_muskits.sh
	$(call end_timer_and_touch,muskits.done)

cauchy_mult.done: espnet.done
	$(call start_timer,cauchy_mult.done)
	. ./activate_python.sh && ./installers/install_cauchy_mult.sh
	$(call end_timer_and_touch,cauchy_mult.done)

rvad_fast.done: espnet.done
	$(call start_timer,rvad_fast.done)
	git clone https://github.com/zhenghuatan/rVADfast.git
	$(call end_timer_and_touch,rvad_fast.done)

discrete_speech_metrics.done: espnet.done
	$(call start_timer,discrete_speech_metrics.done)
	. ./activate_python.sh && ./installers/install_discrete_speech_metrics.sh
	$(call end_timer_and_touch,discrete_speech_metrics.done)

versa.done: espnet.done
	$(call start_timer,versa.done)
	. ./activate_python.sh && ./installers/install_versa.sh
	$(call end_timer_and_touch,versa.done)

check_install: python
	. ./activate_python.sh; . ./extra_path.sh; python3 check_install.py

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
