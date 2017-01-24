# build.py
import os
import platform
import sys
from distutils.core import setup

from torch.utils.ffi import create_extension

extra_compile_args = ['-std=c++11', '-fPIC']
warp_ctc_path = "../build"

if platform.system() == 'Darwin':
    lib_ext = ".dylib"
else:
    lib_ext = ".so"

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
    headers=['src/binding.h'],
    sources=['src/binding.cpp'],
    with_cuda=True,
    include_dirs=include_dirs,
    library_dirs=[os.path.realpath(warp_ctc_path)],
    runtime_library_dirs=[os.path.realpath(warp_ctc_path)],
    libraries=['warpctc'],
    extra_compile_args=extra_compile_args)
ffi = ffi.distutils_extension()
ffi.name = 'warpctc_pytorch._warp_ctc'
setup(
    name="warpctc_pytorch",
    version="0.1",
    packages=["warpctc_pytorch"],
    ext_modules=[ffi],
)
