#!/usr/bin/env python
from distutils.version import LooseVersion
import os
import pip
from setuptools import find_packages
from setuptools import setup
import sys


if LooseVersion(sys.version) < LooseVersion('3.6'):
    raise RuntimeError(
        'ESPnet requires Python>=3.6, '
        'but your Python is {}'.format(sys.version))
if LooseVersion(pip.__version__) < LooseVersion('19'):
    raise RuntimeError(
        'pip>=19.0.0 is required, but your pip is {}. '
        'Try again after "pip install -U pip"'.format(pip.__version__))

requirements = {
    'install': [
        # for some reason, matplotlib should be separately installed
        # 'matplotlib',
        'scipy',
        'h5py',
        # Installation from anaconda is recommended for PyTorch
        # 'torch==0.4.1',
        'chainer==6.0.0',
        # 'cupy==5.0.0',
        'python_speech_features>=0.6',
        'setuptools>=38.5.1',
        'scikit-learn==0.20',
        'librosa>=0.6.2',
        'soundfile>=0.10.2',
        'inflect>=1.0.0',
        'unidecode>=1.0.22',
        'editdistance==0.5.2',
        'h5py>=2.9.0',
        'tensorboardX>=1.4',
        'pillow>=5.3.0',
        'nara_wpe',
        'museval',
        'pystoi',
        'kaldiio',
        # A backport of inspect.signature for python2
        'funcsigs',
        'configargparse',
        'PyYAML',
        'torch_complex@git+https://github.com/kamo-naoyuki/pytorch_complex.git',
        'pytorch_wpe@git+https://github.com/nttcslab-sp/dnn_wpe.git',
    ],
    'setup': ['numpy', 'pytest-runner'],
    'test': [
        'pytest>=3.3.0',
        'pytest-pythonpath>=0.7.3',
        'pytest-cov>=2.7.1',
        'hacking>=1.1.0',
        'mock>=2.0.0',
        'autopep8>=1.3.3',
        'jsondiff'],
    'doc': [
        'Sphinx==1.7.4',
        'sphinx-rtd-theme>=0.2.4',
        'sphinx-argparse>=0.2.5',
        'commonmark==0.8.1',
        'recommonmark>=0.4.0',
        'travis-sphinx>=2.0.1']}
install_requires = requirements['install']
setup_requires = requirements['setup']
tests_require = requirements['test']
extras_require = {k: v for k, v in requirements.items()
                  if k not in ['install', 'setup']}

dirname = os.path.dirname(__file__)
setup(name='espnet',
      version='0.4.0',
      url='http://github.com/espnet/espnet',
      author='Shinji Watanabe',
      author_email='shinjiw@ieee.org',
      description='ESPnet: end-to-end speech processing toolkit',
      long_description=open(os.path.join(dirname, 'README.md'),
                            encoding='utf-8').read(),
      license='Apache Software License',
      packages=find_packages(include=['espnet*']),
      # #448: "scripts" is inconvenient for developping because they are copied
      # scripts=get_all_scripts('espnet/bin'),
      install_requires=install_requires,
      setup_requires=setup_requires,
      tests_require=tests_require,
      extras_require=extras_require,
      classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'Operating System :: POSIX :: Linux',
          'License :: OSI Approved :: Apache Software License',
          'Topic :: Software Development :: Libraries :: Python Modules'],
      )
