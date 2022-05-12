#!/usr/bin/env bash
set -euo pipefail
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $# -ne 1 ]; then
    log "Usage: $0 <cuda_version>"
    exit 1
else
    cuda_version="$1"
fi
if [ "${cuda_version}" = cpu ] || [ "${cuda_version}" = CPU ]; then
    cuda_version=
fi


# espnet requires chiner=6.0.0
chainer_version=6.0.0
python_version=$(python3 -c "import sys; print(sys.version.split()[0])")
cuda_version_without_dot="${cuda_version/\./}"
python_plus(){
    python3 <<EOF
from packaging.version import parse as L
if L('$python_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}
cuda_plus(){
    python3 <<EOF
from packaging.version import parse as L
if L('$cuda_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}




if [ ! -d chainer/ ]; then
    git clone --depth 1 https://github.com/chainer/chainer -b v${chainer_version}
fi

if [ ! -e chainer/.patched ]; then

    # Remove typing and make the protobuf-requirements relaxing
    cat <<EOF | patch -u chainer/setup.py
diff --git a b
index 269bff4..cee3f71 100644
--- a
+++ b
@@ -24,7 +24,6 @@ set CHAINER_PYTHON_350_FORCE environment variable to 1."""
 requirements = {
     'install': [
         'setuptools',
-        'typing',
         'typing_extensions',
         'filelock',
         'numpy>=1.9.0',
@@ -32,7 +31,7 @@ requirements = {
         # TODO(niboshi): Probably we should always use pip in CIs for
         # installing chainer. It avoids pre-release dependencies by default.
         # See also: https://github.com/pypa/setuptools/issues/855
-        'protobuf>=3.0.0,<3.8.0rc1',
+        'protobuf',
         'six>=1.9.0',
     ],
     'stylecheck': [
EOF
    touch chainer/.patched
fi


python3 -m pip install -e chainer/


# CUPY installation
if [ -n "${cuda_version}" ]; then
    if $(cuda_plus 10.2); then
        echo "[INFO] Skip cupy installation"
    else
        if $(python_plus 3.8); then
            python3 -m pip install "cupy==${chainer_version}"
        else
            python3 -m pip install "cupy-cuda${cuda_version_without_dot}==${chainer_version}"
        fi
    fi
fi
