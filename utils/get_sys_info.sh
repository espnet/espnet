#!/bin/bash

# Copyright 2019 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

[ -f ./path.sh ] && . ./path.sh

. utils/parse_options.sh

echo "  - Environments (obtained by \`\$ get_sys_info.sh\`)"

# get date info
echo -n "    - date: \`"
date | sed -e "s/$/\`/" 
# get system info
echo -n "    - system information: \`"
uname -a | sed -e "s/$/\`/" 
# get python version
echo -n "    - python version: \`"
python --version | sed -e "s/$/\`/"
# get espnet version
echo -n "    - espnet version: \`"
python -c 'import espnet; print("espnet " + espnet.__version__);' | sed -e "s/$/\`/"
# get chainer version
echo -n "    - chainer version: \`"
python -c 'import chainer; print("chainer " + chainer.__version__);' | sed -e "s/$/\`/"
# get pytorch version
echo -n "    - pytorch version: \`"
python -c 'import torch; print("pytorch " + torch.__version__);' | sed -e "s/$/\`/"

# get Git hash
echo -n "    - Git hash: \`"
git log | head -n 1 | awk '{print $2}' | sed -e "s/$/\`/"

exit 0
