import numpy
from setuptools import setup, Extension

setup(
    name="chainer_ctc",
    version="1.0",
    description="Fast CTC for Chainer + wrapper for Baidus warp-ctc",
    platforms=['Linux'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
    ],
    author="Jahn Heymann",
    author_email="jahnheymann@gmail.com",
    setup_requires=['setuptools_cython', 'Cython >= 0.18'],
    ext_modules=[
        Extension('chainer_ctc.src.ctc_cpu', ["chainer_ctc/src/ctc_cpu.pyx"],
                  include_dirs=[numpy.get_include()],
                  language='c++'),
        Extension('chainer_ctc.src.warp_ctc',
                  ["chainer_ctc/src/warp_ctc.pyx"],
                  include=[numpy.get_include(),
                           'ctc.h'],
                  include_dirs=['ext/warp-ctc/include'],
                  library_dirs=['ext/warp-ctc/build'],
                  libraries=['warpctc'],
                  language='c++')
    ]
)
