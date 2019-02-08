PYTHON := /usr/bin/python2.7
KALDI =

.PHONY: all clean

all: venv chainer_patch.done kaldi.done warp-ctc.done chainer_ctc.done kaldiio extra

all_python: venv chainer_patch.done warp-ctc.done chainer_ctc.done

extra: nkf.done sentencepiece.done mecab.done moses

ifneq ($(strip $(KALDI)),)
kaldi.done:
	ln -s $(KALDI) kaldi
	touch kaldi.done
else
kaldi.done:
	test -d kaldi || git clone https://github.com/kaldi-asr/kaldi.git
	cd kaldi/tools; $(MAKE) all
	cd kaldi/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
	touch kaldi.done
endif

venv:
	test -d venv || virtualenv -p $(PYTHON) venv

espnet.done: venv
	. venv/bin/activate; pip install pip --upgrade
	. venv/bin/activate; pip install -e ..
	. venv/bin/activate; pip install cupy==5.0.0 torch==0.4.1 matplotlib
	touch espnet.done

kaldiio: venv
	. venv/bin/activate; pip install git+https://github.com/nttcslab-sp/kaldiio.git

chainer_patch.done: espnet.done
	$(eval FILENAME := $(shell find venv/lib -name "multiprocess_iterator.py"))
	. venv/bin/activate; patch $(FILENAME) < prefetch.patch
	touch chainer_patch.done

warp-ctc.done: espnet.done
	rm -rf warp-ctc
	git clone https://github.com/jnishi/warp-ctc.git
	. venv/bin/activate; cd warp-ctc && mkdir build && cd build && cmake .. && make -j4 ; true
	. venv/bin/activate; pip install cffi
	. venv/bin/activate; cd warp-ctc/pytorch_binding && python setup.py install # maybe need to: apt-get install python-dev
	touch warp-ctc.done

chainer_ctc.done: espnet.done
	rm -rf chainer_ctc
	git clone https://github.com/jheymann85/chainer_ctc.git
	. venv/bin/activate; pip install cython
	. venv/bin/activate; cd chainer_ctc && chmod +x install_warp-ctc.sh && ./install_warp-ctc.sh ; true
	. venv/bin/activate; cd chainer_ctc && pip install .
	touch chainer_ctc.done

nkf.done:
	rm -rf nkf
	mkdir -p nkf
	cd nkf; wget https://ja.osdn.net/dl/nkf/nkf-2.1.4.tar.gz
	cd nkf; tar zxvf nkf-2.1.4.tar.gz; cd nkf-2.1.4; $(MAKE) prefix=.
	touch nkf.done

sentencepiece.done:
	rm -rf sentencepiece
	git clone https://github.com/google/sentencepiece.git
	cd sentencepiece && mkdir build && cd build && (cmake3 .. || cmake ..) && $(MAKE)
	touch sentencepiece.done

mecab.done:
	rm -rf mecab
	git clone https://github.com/taku910/mecab.git
	cd mecab/mecab && ./configure --with-charset=utf8 --prefix=$(PWD)/mecab && $(MAKE) && $(MAKE) check && $(MAKE) install
	cd mecab/mecab-ipadic && ./configure --with-charset=utf8 --with-mecab-config=$(PWD)/mecab/bin/mecab-config --prefix=$(PWD)/mecab && $(MAKE) && $(MAKE) install
	cd mecab && git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
	cd mecab/mecab-ipadic-neologd && export PATH=$(PWD)/mecab/bin:$(PATH) && ./bin/install-mecab-ipadic-neologd -n -y -p $(PWD)/mecab/mecab-ipadic-neologd
	. venv/bin/activate; pip install mojimoji pykakasi
	. venv/bin/activate; if [ `python --version | cut -c 8` -eq 3 ]; then \
		if [ ! -e swig.done ]; then \
			rm -rf swig; \
			mkdir -p swig; \
			cd swig; \
			wget https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/swig-3.0.12.tar.gz; \
			tar zxvf swig-3.0.12.tar.gz; \
			cd swig-3.0.12 && ./configure prefix=$(PWD)/swig && $(MAKE) && $(MAKE) install; \
			touch $(PWD)/swig.done; \
		fi; \
		deactivate; \
		export PATH=$(PWD)/swig/bin:$(PATH); \
		. venv/bin/activate; pip install mecab-python3; \
	else \
		pip install mecab-python; \
	fi
	touch mecab.done

moses:
	git clone https://github.com/moses-smt/mosesdecoder.git moses

clean: clean_extra
	rm -rf kaldi venv warp-ctc chainer_ctc
	rm -f miniconda.sh
	rm -rf *.done
	find . -iname "*.pyc" -delete

clean_python:
	rm -rf venv warp-ctc chainer_ctc kaldi-io-for-python
	rm -f miniconda.sh
	rm -rf chainer_patch.done warp-ctc.done chainer_ctc.done
	find . -iname "*.pyc" -delete

clean_extra:
	rm -rf nkf sentencepiece mecab swig moses
