# build.py
import os
import platform
import sys
from setuptools import setup

from torch.utils.ffi import create_extension

extra_compile_args = ['-std=c++11', '-fPIC']
warp_ctc_path = "../build"

if "CUDA_HOME" not in os.environ:
    print("CUDA_HOME not found in the environment so building "
          "without GPU support. To build with GPU support "
          "please define the CUDA_HOME environment variable. "
          "This should be a path which contains include/cuda.h")
    enable_gpu = False
else:
    enable_gpu = True

if platform.system() == 'Darwin':
    lib_ext = ".dylib"
else:
    lib_ext = ".so"

headers = ['src/cpu_binding.h']

if enable_gpu:
    extra_compile_args += ['-DWARPCTC_ENABLE_GPU']
    headers += ['src/gpu_binding.h']

if "WARP_CTC_PATH" in os.environ:
    warp_ctc_path = os.environ["WARP_CTC_PATH"]
if not os.path.exists(os.path.join(warp_ctc_path, "libwarpctc" + lib_ext)):
    print(("Could not find libwarpctc.so in {}.\n"
           "Build warp-ctc and set WARP_CTC_PATH to the location of"
           " libwarpctc.so (default is '../build')").format(warp_ctc_path))
    sys.exit(1)
include_dirs = [os.path.realpath('../include')]

ffi = create_extension(
    name='warp_ctc',
    language='c++',
    headers=headers,
    sources=['src/binding.cpp'],
    with_cuda=enable_gpu,
    include_dirs=include_dirs,
    library_dirs=[os.path.realpath(warp_ctc_path)],
    libraries=['warpctc'],
    extra_link_args=['-Wl,-rpath,' + os.path.realpath(warp_ctc_path)],
    extra_compile_args=extra_compile_args)
ffi = ffi.distutils_extension()
ffi.name = 'warpctc_pytorch._warp_ctc'
setup(
    name="warpctc_pytorch",
    version="0.1",
    description="PyTorch wrapper for warp-ctc",
    url="https://github.com/baidu-research/warp-ctc",
    author="Jared Casper, Sean Naren",
    author_email="jared.casper@baidu.com, sean.narenthiran@digitalreasoning.com",
    license="Apache",
    packages=["warpctc_pytorch"],
    ext_modules=[ffi],
)
