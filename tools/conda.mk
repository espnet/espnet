PYTHON_VERSION := 2.7
PATH := $(PWD)/venv/bin:$(PATH)

include Makefile

ifeq ($(PYTHON_VERSION),2.7)
	CONDA_URL = https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
else
	CONDA_URL = https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
endif

.PHONY: all clean

miniconda.sh:
	test -f miniconda.sh || wget $(CONDA_URL) -O miniconda.sh

conda: miniconda.sh
	test -d $(PWD)/venv || bash miniconda.sh -b -p $(PWD)/venv
	conda config --set always_yes yes --set changeps1 no
	conda update conda
	conda install python=$(PYTHON_VERSION)
	conda info -a

virtualenv: conda requirements.txt
	. venv/bin/activate && conda install -y numpy matplotlib
	. venv/bin/activate && grep -v torch requirements.txt | pip install -r /dev/stdin
	. venv/bin/activate && conda install pytorch -c pytorch
