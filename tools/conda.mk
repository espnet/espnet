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

venv: miniconda.sh
	test -d $(PWD)/venv || bash miniconda.sh -b -p $(PWD)/venv

espnet.done: venv
	conda config --set always_yes yes --set changeps1 no
	conda update conda
	conda install python=$(PYTHON_VERSION)
	conda info -a
	. venv/bin/activate && conda install -y pytorch -c pytorch
	. venv/bin/activate && conda install -y hp5y matplotlib
	. venv/bin/activate && pip install -e ..
	. venv/bin/activate && pip install cupy==4.3.0
	touch espnet.done
