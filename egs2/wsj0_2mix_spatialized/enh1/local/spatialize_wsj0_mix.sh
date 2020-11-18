#!/bin/bash

# Copyright  2020  Shanghai Jiao Tong University (Author: Wangyou Zhang)
# Apache 2.0

min_or_max=min
sample_rate=8000
nj=16

. utils/parse_options.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [[ "$min_or_max" != "max" ]] && [[ "$min_or_max" != "min" ]]; then
  echo "Error: min_or_max must be either max or min: ${min_or_max}"
  exit 1
fi
if [[ "$sample_rate" == "16k" ]]; then
  sample_rate=16000
elif [[ "$sample_rate" == "8k" ]]; then
  sample_rate=8000
else
  echo "Error: sample rate must be either 16k or 8k: ${sample_rate}"
  exit 1
fi

if [ $# -ne 3 ]; then
  echo "Usage: $0 <dir> <wsj0-2mix-wav> <wsj0-2mix-spatialized-wav>"
  echo " where <dir> is download space,"
  echo " <wsj0-2mix-wav> is the generated wsj0-2mix path,"
  echo " <wsj0-2mix-spatialized-wav> is the wav generation space."
  echo "Note: this script won't actually re-download things if called twice,"
  echo "because we use the --continue flag to 'wget'."
  echo "Note: this script can be used to create spatialized wsj0_2mix corpus"
  exit 1;
fi

dir=$1
wsj0_2mix_wav=$3
wsj0_2mix_spatialized_wav=$2


if ! command -v matlab >/dev/null 2>&1; then
    echo "matlab not found."
    exit 1
fi

if ! command -v mex >/dev/null 2>&1; then
    echo "mex not found."
    exit 1
fi

rootdir=$PWD
echo "Downloading spatialize_WSJ0_mixture scripts."

url=https://www.merl.com/demos/deep-clustering/spatialize_wsj0-mix.zip
wdir=data/local/downloads
url2=https://raw.githubusercontent.com/ehabets/RIR-Generator/master/rir_generator.cpp

mkdir -p ${dir}
mkdir -p ${dir}/RIR-Generator-master
mkdir -p ${wdir}/log

# Download and modiy spatialize_wsj0 scripts
wget --continue -O $wdir/spatialize_wsj0-mix.zip ${url}

unzip ${wdir}/spatialize_wsj0-mix.zip -d ${dir}

sed -i -e "s#data_in_root  = './wsj0-mix/';#data_in_root  = '${wsj0_2mix_wav}';#" \
       -e "s#rir_root      = './wsj0-mix/';#rir_root      = '${wsj0_2mix_spatialized_wav}';#" \
       -e "s#data_out_root = './wsj0-mix/';#data_out_root = '${wsj0_2mix_spatialized_wav}';#" \
       ${dir}/spatialize_wsj0_mix.m

sed -i -e "s#MIN_OR_MAX=\"'min'\"#MIN_OR_MAX=\"'${min_or_max}'\"#" \
       -e "s#FS=8000#FS=${sample_rate}#" \
       -e "s#NUM_JOBS=20#NUM_JOBS=${nj}#" \
       ${dir}/launch_spatialize.sh

# Download and compile rir_generator
wget --continue -O ${dir}/RIR-Generator-master/rir_generator.cpp ${url2}
(cd ${dir}/RIR-Generator-master && mex rir_generator.cpp)

echo "Spatializing Mixtures."
NUM_SPEAKERS=2
MIN_OR_MAX="'${min_or_max}'"
FS=${sample_rate}         # 16000 or 8000
START_IND=1
STOP_IND=28000            # number of utts: 20000+5000+3000
USEPARCLUSTER_WITH_IND=1  # 1 for using parallel processing toolbox
GENERATE_RIRS=1           # 1 for generating RIRs

NUM_WORKERS=${nj}         # maximum of 1 MATLAB worker per CPU core is recommended
sed -i -e "s#c.NumWorkers = 22;#c.NumWorkers = ${NUM_WORKERS};#" ${dir}/spatialize_wsj0_mix.m

# Java must be initialized in order to use the Parallel Computing Toolbox.
# Please launch MATLAB without the '-nojvm' flag.
matlab_cmd="matlab -nodesktop -nodisplay -nosplash -r \"spatialize_wsj0_mix(${NUM_SPEAKERS},${MIN_OR_MAX},${FS},${START_IND},${STOP_IND},${USEPARCLUSTER_WITH_IND},${GENERATE_RIRS})\""

cmdfile=${dir}/spatialize_matlab.sh
echo "#!/bin/bash" > $cmdfile
echo $matlab_cmd >> $cmdfile
chmod +x $cmdfile

# Run Matlab (This takes more than 8 hours)
# Expected data directory to be generated:
#   - ${wsj0_2mix_spatialized_wav}/RIRs_16k/rir_*.mat
#   - ${wsj0_2mix_spatialized_wav}/2speakers_anechoic/wav16k/${min_or_max}/{tr,cv,tt}/{mix,s1,s2}/*.wav
#   - ${wsj0_2mix_spatialized_wav}/2speakers_reverb/wav16k/${min_or_max}/{tr,cv,tt}/{mix,s1,s2}/*.wav
cd ${dir}
echo "Log is in ${dir}/spatialize.log"
$train_cmd ${dir}/spatialize.log $cmdfile

cd ${rootdir}
