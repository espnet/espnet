from setuptools import setup, find_packages
from os.path import join, dirname
import stress_dictionary as sd

setup(
    name='stress-dictionary',
    version=sd.__version__,
    packages=find_packages(),
    package_data={'stress-dictionary': ['dicts/wiki-stresses.json']},
    description='Set stresses in English/Russian texts',
    long_description=open(join(dirname(__file__), 'README.md')).read(),
    long_description_content_type='text/markdown',
    classifiers='Topic :: Text Processing, Programming Language :: Python :: 3.7',
    license='Private License TWIN 1.0',
    entry_points={ 'console_scripts': [
            'stresses_dictionary = stress_dictionary.stresses_dictionary:main',
            'stresses_utils = stress_dictionary.stresses_utils:main'
        ]},
    install_requires=['docopt', 'prompt_toolkit'],
    author='Dmitry Krylov',
    author_email='dima@krylov.ws',
    url='https://twin24.ai',
    keywords='stress,dictionary',
    project_urls={'Source Code': 'https://gitlab.twin24.ai/ai/support/stress_dictionary'},
    include_package_data=True,
)
