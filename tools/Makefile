### custom flags ###

# for python3.6: make PYTHON_VERSION=3.6
PYTHON_VERSION := 2.7
# for cpu only: make PYTORCH=pytorch-cpu
# for CUDA9.1:  make PYTORCH="pytorch cuda91"
PYTORCH := pytorch
# for your prebuilt kaldi: make KALDI_ROOT=<prebuilt kaldi>
KALDI_ROOT := kaldi_github

### auto flags ###
KALDI_BIN := $(KALDI_ROOT)/src/feat/libkaldi-feat.so
BEAMFORMIT := $(KALDI_ROOT)/tools/BeamformIt/BeamformIt
ACT := venv/bin/activate
SITE := "venv/lib/python$(PYTHON_VERSION)/site-packages"
ifeq ($(PYTHON_VERSION),2.7)
	CONDA_SCRIPT := Miniconda2-latest-Linux-x86_64.sh
else
	CONDA_SCRIPT := Miniconda3-latest-Linux-x86_64.sh
endif
LD_LIBRARY_PATH := "$(PWD)/venv/lib:$(LD_LIBRARY_PATH)"
LIBRARY_PATH := "$(PWD)/venv/lib:$(LIBRARY_PATH)"
WARPCTC_INIT := "warp-ctc/pytorch_binding/build/lib.linux-x86_64-$(PYTHON_VERSION)/warpctc_pytorch/__init__.py"

.PHONY: all clean test

all: kaldi nkf kaldi-io-for-python $(WARPCTC_INIT) $(SITE)/chainer_ctc subword-nmt


### kaldi related ###
kaldi-io-for-python:
	git clone https://github.com/vesis84/kaldi-io-for-python.git
	cd ../src/utils; ln -s ../../tools/kaldi-io-for-python/kaldi_io.py kaldi_io_py.py

kaldi: $(KALDI_BIN) $(BEAMFORMIT)
	ln -s $(KALDI_ROOT) kaldi

$(BEAMFORMIT): $(KALDI_ROOT) $(ACT)
	. $(ACT) && cd $(KALDI_ROOT)/tools; $(MAKE) all && ./extras/install_beamformit.sh
	ls $(BEAMFORMIT)

$(KALDI_BIN): $(KALDI_ROOT) $(ACT)
	. $(ACT) && cd $(KALDI_ROOT)/src; ./configure --shared --use-cuda=no --mathlib=MKL --mkl-libdir=../../venv/lib --mkl-root=../../venv; $(MAKE) depend; $(MAKE) all
	ls $(KALDI_BIN)

kaldi_github:
	git clone https://github.com/kaldi-asr/kaldi.git --depth 1 kaldi_github


### pip/conda related ###
$(CONDA_SCRIPT):
	wget https://repo.continuum.io/miniconda/$(CONDA_SCRIPT)

$(ACT): $(CONDA_SCRIPT)
	bash $(CONDA_SCRIPT) -b -p $(PWD)/venv
	. $(ACT) && conda config --set always_yes yes
	. $(ACT) && conda update conda
	. $(ACT) && conda info -a
	. $(ACT) && conda install mkl-include matplotlib $(PYTORCH) -c pytorch
	. $(ACT) && pip install -U pip
	. $(ACT) && which nvcc && pip install "cupy>=2.0.0" || echo "skip cupy"
	. $(ACT) && pip install -r requirements.txt


### python packages out of pip/conda ###
warp-ctc:
	git clone https://github.com/SeanNaren/warp-ctc.git
	cd warp-ctc && git checkout 9e5b238f8d9337b0c39b3fd01bbaff98ba523aa5

warp-ctc/build/libwarpctc.so: warp-ctc
	. $(ACT) && cd warp-ctc && rm -rf build && mkdir build && cd build && cmake .. && make -j4

$(WARPCTC_INIT): $(ACT) warp-ctc/build/libwarpctc.so
	. $(ACT) && cd warp-ctc/pytorch_binding && python setup.py install

chainer_ctc:
	git clone https://github.com/jheymann85/chainer_ctc.git --depth 1

$(SITE)/chainer_ctc: $(ACT) chainer_ctc
	. $(ACT) && cd chainer_ctc && chmod +x install_warp-ctc.sh && ./install_warp-ctc.sh && pip install .

subword-nmt:
	git clone https://github.com/rsennrich/subword-nmt.git --depth 1


### misc ###
nkf:
	mkdir -p nkf
	cd nkf; wget http://gigenet.dl.osdn.jp/nkf/64158/nkf-2.1.4.tar.gz
	cd nkf; tar zxvf nkf-2.1.4.tar.gz; cd nkf-2.1.4; $(MAKE) prefix=.


clean:
	rm -fr kaldi_github kaldi kaldi_python venv nkf kaldi-io-for-python ../src/utils/kaldi_io_py.py warp-ctc chainer_ctc subword-nmt
	rm -f Miniconda*.sh
	rm -f chainer300_cudnn7.0_nossh.devel
	find . -iname "*.pyc" -delete


## test ##
venv/bin/pytest: $(ACT)
	. $(ACT) && conda install pytest && pip install -r test_requirements.txt

test: venv/bin/pytest
	. $(ACT) && cd .. && pytest
	. $(ACT) && cd .. && flake8 src test
	. $(ACT) && cd .. && autopep8 -r src test --exclude src/utils --global-config .pep8 --diff --max-line-length 120 | tee check_autopep8
	. $(ACT) && cd .. && test ! -s check_autopep8

