# check extra module installation
torchaudio_version=$(python3 -c "import torchaudio; print(torchaudio.__version__)")

torchaudio_plus(){
    python3 <<EOF
from packaging.version import parse as L
if L('$torchaudio_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}

if ! $(torchaudio_plus 0.13.1); then
    echo "We recommend to run hubert with torchaudio version >= 0.13.1." >$2
fi

if ! python3 -c "import fairseq" > /dev/null; then
    echo "Warning: fairseq is not installed." >&2
    echo "Warning: ignore this warning if torchaudio hubert config is used." >&2
    echo "Warning: please install fairseq and its dependencies as follows:" >&2
    echo "Warning: cd ${MAIN_ROOT}/tools && make fairseq.done" >&2
    return 1
fi