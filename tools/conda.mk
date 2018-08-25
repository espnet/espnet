PYTHON_VERSION := 2.7
PATH := $(PWD)/venv/bin:$(PATH)

include Makefile

ifeq ($(PYTHON_VERSION),2.7)
	CONDA_URL = https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
else
	CONDA_URL = https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
endif

.PHONY: all clean pytorch venv/lib/python2.7/site-packages/torch

miniconda.sh:
	wget $(CONDA_URL) -O miniconda.sh

venv: miniconda.sh
	bash miniconda.sh -b -p $(PWD)/$@
	conda config --set always_yes yes --set changeps1 no
	conda update conda
	conda info -a

venv/bin/activate: venv requirements.txt
	. venv/bin/activate && conda install -y numpy matplotlib && pip install -r requirements.txt

pytorch: venv/bin/activate
	. venv/bin/activate && conda install pytorch==0.3.1 -c pytorch

venv/lib/python2.7/site-packages/torch: pytorch
	echo "we pass this rule using pip because we use conda"
