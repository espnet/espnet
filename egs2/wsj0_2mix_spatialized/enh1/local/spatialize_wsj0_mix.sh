#!/usr/bin/env bash

# Copyright  2020  Shanghai Jiao Tong University (Author: Wangyou Zhang)
# Apache 2.0

min_or_max=min
sample_rate=8k
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
wsj0_2mix_wav=$2
wsj0_2mix_spatialized_wav=$3


if ! command -v matlab >/dev/null 2>&1; then
    echo "matlab not found."
    exit 1
fi

if ! command -v mex >/dev/null 2>&1; then
    echo "mex not found."
    exit 1
fi

echo "Downloading spatialize_WSJ0_mixture scripts."

url=https://www.merl.com/demos/deep-clustering/spatialize_wsj0-mix.zip
wdir=data/local/downloads

mkdir -p ${dir}
mkdir -p ${wdir}/log

# Download and modiy spatialize_wsj0 scripts
wget --continue -O $wdir/spatialize_wsj0-mix.zip ${url}

unzip ${wdir}/spatialize_wsj0-mix.zip -d ${dir}

sed -i -e "s#data_in_root  = './wsj0-mix/';#data_in_root  = '${wsj0_2mix_wav}';#" \
       -e "s#rir_root      = './wsj0-mix/';#rir_root      = '${wsj0_2mix_spatialized_wav}';#" \
       -e "s#data_out_root = './wsj0-mix/';#data_out_root = '${wsj0_2mix_spatialized_wav}';#" \
       -e "s#RIR-Generator-master/#RIR-Generator/#" \
       ${dir}/spatialize_wsj0_mix.m

sed -i -e "s#MIN_OR_MAX=\"'min'\"#MIN_OR_MAX=\"'${min_or_max}'\"#" \
       -e "s#FS=8000#FS=${sample_rate}#" \
       -e "s#NUM_JOBS=20#NUM_JOBS=${nj}#" \
       ${dir}/launch_spatialize.sh

# Download and compile rir_generator
git clone https://github.com/ehabets/RIR-Generator "${dir}/RIR-Generator"
(cd "${dir}/RIR-Generator" && mex rir_generator.cpp rir_generator_core.cpp)
rir_generator=$(realpath ${dir}/RIR-Generator/rir_generator.mexa64)

echo "Spatializing Mixtures."
NUM_SPEAKERS=2
MIN_OR_MAX="'${min_or_max}'"
FS=${sample_rate}         # 16000 or 8000
START_IND=1
STOP_IND=28000            # number of utts: 20000+5000+3000
USEPARCLUSTER_WITH_IND=1  # 1 for using parallel processing toolbox
GENERATE_RIRS=1           # 1 for generating RIRs

NUM_WORKERS=${nj}         # maximum of 1 MATLAB worker per CPU core is recommended
sed -i -e "s#c.NumWorkers = 22;#c.NumWorkers = ${NUM_WORKERS};#" \
    -e "/parpool(c, c.NumWorkers);/a addAttachedFiles(gcp, {'${rir_generator}'});" \
    ${dir}/spatialize_wsj0_mix.m

# Java must be initialized in order to use the Parallel Computing Toolbox.
# Please launch MATLAB without the '-nojvm' flag.
matlab_cmd="matlab -nodesktop -nodisplay -nosplash -r \"spatialize_wsj0_mix(${NUM_SPEAKERS},${MIN_OR_MAX},${FS},${START_IND},${STOP_IND},${USEPARCLUSTER_WITH_IND},${GENERATE_RIRS})\""

cmdfile=${dir}/spatialize_matlab.sh
echo "#!/usr/bin/env bash" > $cmdfile
echo "cd ${dir}" >> $cmdfile
echo $matlab_cmd >> $cmdfile
chmod +x $cmdfile

# Run Matlab (This takes more than 8 hours)
# Expected data directories to be generated:
#   - ${wsj0_2mix_spatialized_wav}/RIRs_16k/rir_*.mat
#   - ${wsj0_2mix_spatialized_wav}/2speakers_anechoic/wav16k/${min_or_max}/{tr,cv,tt}/{mix,s1,s2}/*.wav
#   - ${wsj0_2mix_spatialized_wav}/2speakers_reverb/wav16k/${min_or_max}/{tr,cv,tt}/{mix,s1,s2}/*.wav
# -----------------------------------------------------------------------------------------
# directory (same for 2speakers_reverb)   disk usage  duration      #samples
# -----------------------------------------------------------------------------------------
# 2speakers_anechoic/wav16k/max/tr/mix    41  GiB     46h 56m 16s   20000 * 8 (8 channels)
# 2speakers_anechoic/wav16k/max/tr/s1     34  GiB     46h 56m 16s   20000 * 8 (8 channels)
# 2speakers_anechoic/wav16k/max/tr/s2     33  GiB     46h 56m 16s   20000 * 8 (8 channels)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2speakers_anechoic/wav16k/max/cv/mix    11  GiB     11h 53m 36s   5000 * 8 (8 channels)
# 2speakers_anechoic/wav16k/max/cv/s1     8.5 GiB     11h 53m 36s   5000 * 8 (8 channels)
# 2speakers_anechoic/wav16k/max/cv/s2     8.4 GiB     11h 53m 36s   5000 * 8 (8 channels)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2speakers_anechoic/wav16k/max/tt/mix    6.2 GiB     7h 20m 48s    3000 * 8 (8 channels)
# 2speakers_anechoic/wav16k/max/tt/s1     5.1 GiB     7h 20m 48s    3000 * 8 (8 channels)
# 2speakers_anechoic/wav16k/max/tt/s2     5.1 GiB     7h 20m 48s    3000 * 8 (8 channels)
# -----------------------------------------------------------------------------------------
# 2speakers_anechoic/wav8k/min/tr/mix     27 GiB      30h 22m 49s   20000 * 8 (8 channels)
# 2speakers_anechoic/wav8k/min/tr/s1      27 GiB      30h 22m 49s   20000 * 8 (8 channels)
# 2speakers_anechoic/wav8k/min/tr/s2      27 GiB      30h 22m 49s   20000 * 8 (8 channels)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2speakers_anechoic/wav8k/min/cv/mix     6.7 GiB     7h 40m 21s    5000 * 8 (8 channels)
# 2speakers_anechoic/wav8k/min/cv/s1      6.7 GiB     7h 40m 21s    5000 * 8 (8 channels)
# 2speakers_anechoic/wav8k/min/cv/s2      6.6 GiB     7h 40m 21s    5000 * 8 (8 channels)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2speakers_anechoic/wav8k/min/tt/mix     4.1 GiB     4h 49m 33s    3000 * 8 (8 channels)
# 2speakers_anechoic/wav8k/min/tt/s1      4.2 GiB     4h 49m 33s    3000 * 8 (8 channels)
# 2speakers_anechoic/wav8k/min/tt/s2      4.1 GiB     4h 49m 33s    3000 * 8 (8 channels)
# -----------------------------------------------------------------------------------------
echo "Log is in ${dir}/spatialize.log"
$train_cmd ${dir}/spatialize.log $cmdfile
