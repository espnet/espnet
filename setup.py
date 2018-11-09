#!/usr/bin/env python
import os
from setuptools import find_packages
from setuptools import setup


def get_all_scripts(dirname):
    return [os.path.join(root, f)
            for root, _, files in os.walk(dirname) for f in files
            if os.path.splitext(f)[1] == '.py' and f != '__init__.py']


requirements = {
    'install': [
        # for some reason, matplotlib should be separately installed
        # 'matplotlib',
        'scipy',
        # Installation from anaconda is recommended for PyTorch
        # 'torch==0.4.1',
        'chainer==4.3.1',
        # 'cupy==4.3.0',
        'python_speech_features>=0.6',
        'setuptools>=38.5.1',
        'librosa>=0.6.2',
        'soundfile>=0.10.2',
        'inflect>=1.0.0',
        'unidecode>=1.0.22',
        'editdistance==0.5.2'],
    'setup': ['numpy', 'pytest-runner'],
    'test': [
        'pytest>=3.3.0',
        'pytest-pythonpath>=0.7.1',
        'hacking>=1.1.0',
        'mock>=2.0.0',
        'autopep8>=1.3.3'],
    'doc': [
        'Sphinx==1.7.4',
        'sphinx-rtd-theme>=0.2.4',
        'commonmark==0.5.4',
        'recommonmark>=0.4.0',
        'travis-sphinx>=2.0.1']}
install_requires = requirements['install']
setup_requires = requirements['setup']
tests_require = requirements['test']
extras_require = {k: v for k, v in requirements.items()
                  if k not in ['install', 'setup']}

dirname = os.path.dirname(__file__)
setup(name='espnet',
      version='0.2.0',
      url='http://github.com/espnet/espnet',
      author='Shinji Watanabe',
      author_email='shinjiw@ieee.org',
      description='ESPnet: end-to-end speech processing toolkit',
      long_description=open(os.path.join(dirname, 'README.md')).read(),
      license='Apache Software License',
      packages=find_packages(include=['espnet*']),
      scripts=get_all_scripts('espnet/bin'),
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
