---
name: Bug report
about: Create a report to help us improve
title: ''
labels: Bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**Environments (you can obtain them by the following command):**
 - OS information: [e.g., Linux 4.9.0-11-amd64 #1 SMP Debian 4.9.189-3+deb9u2 (2019-11-11) x86_64]
 - python version: [e.g. 3.7.3 (default, Mar 27 2019, 22:11:17) [GCC 7.3.0]]
 - espnet version: [e.g. espnet 0.6.0]
 - pytorch version [e.g. pytorch 1.0.1.post2]
 - Git hash [e.g. 83799e69a0269450587a6857882c73bfb27551d5]
   - Commit date [e.g. Tue Feb 4 14:21:11 2020 +0900]
```
cd <espnet-root>/tools
. ./activate_python.sh

uname -mrsv
python3 << EOF
import sys, espnet, torch
pyversion = sys.version.replace('\n', ' ')
print(f"""- python version: \`{pyversion}\`
- espnet version: \`espnet {espnet.__version__}\`
- pytorch version: \`pytorch {torch.__version__}\`""")
EOF

cat << EOF
- Git hash: \`$(git rev-parse HEAD)\`
  - Commit date: \`$(git log -1 --format='%cd')\`
EOF
```

**Task information:**
 - Task: [e.g., ASR, TTS, ST, ENH]
 - Recipe: [e.g. librispeech]
 - ESPnet1 or ESPnet2

**To Reproduce**
Steps to reproduce the behavior:
1. move to a recipe directory, e.g., `cd egs/librispeech/asr1`
2. execute `run.sh` with specific arguments, e.g., `run.sh --stage 3 --ngp 1`
3. specify the error log, e.g., `exp/xxx/yyy.log`

**Error logs**
Paste the error logs. If applicable, add screenshots to help explain your problem.
