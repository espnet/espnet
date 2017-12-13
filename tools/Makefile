all: kaldi kaldi_python jq venv venv/bin/activate nkf

docker: kaldi kaldi_python_docker jq 

jq:
	mkdir -p jq
	curl -o jq/jq -L https://github.com/stedolan/jq/releases/download/jq-1.5/jq-linux64
	chmod +x -R jq

kaldi_python: kaldi venv
	git clone https://github.com/dmitriy-serdyuk/kaldi-python.git kaldi_python
	. venv/bin/activate; cd kaldi_python; export KALDI_ROOT=`pwd`/../kaldi/; python setup.py install

kaldi_python_docker: kaldi
	git clone https://github.com/dmitriy-serdyuk/kaldi-python.git kaldi_python
	cd kaldi_python; export KALDI_ROOT=`pwd`/../kaldi/; python setup.py install

kaldi: kaldi_github
	cd kaldi_github/tools; git checkout 9e7644cac682123de110ad6580e15f311b953c73; $(MAKE) all
	cd kaldi_github/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
	ln -s kaldi_github kaldi

kaldi_github:
	git clone https://github.com/kaldi-asr/kaldi.git kaldi_github

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv -p /usr/bin/python2.7 venv
	. venv/bin/activate; pip install -r requirements.txt
	touch venv/bin/activate

nkf:
	mkdir -p nkf
	cd nkf; wget http://gigenet.dl.osdn.jp/nkf/64158/nkf-2.1.4.tar.gz
	cd nkf; tar zxvf nkf-2.1.4.tar.gz; cd nkf-2.1.4; $(MAKE) prefix=.

clean:
	rm -fr kaldi_github kaldi kaldi_python jq venv nkf
	find -iname "*.pyc" -delete
