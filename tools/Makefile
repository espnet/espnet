all: kaldi venv venv/bin/activate nkf kaldi-io-for-python.git

kaldi-io-for-python.git:
	git clone https://github.com/vesis84/kaldi-io-for-python.git
	cd ../src/utils; ln -s ../../tools/kaldi-io-for-python/kaldi_io.py kaldi_io_py.py

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
	rm -fr kaldi_github kaldi kaldi_python venv nkf kaldi-io-for-python ../src/utils/kaldi_io_py.py
	find -iname "*.pyc" -delete
