#!/usr/bin/env bash
# エラーが発生した時点でスクリプトを停止する安全装置
set -euo pipefail

# 事前にプロキシ認証スクリプトを通している前提です
~/proxy.sh

MAMBA_ROOT=/data/store5/tkano/mamba_root
ENV_NAME="TH211_CU128"

echo "=== 1. 環境のセットアップと有効化 ==="
#./setup_mamba.sh ${MAMBA_ROOT} ${ENV_NAME} 3.10

# スクリプト内で activate するための初期化
source ${MAMBA_ROOT}/etc/profile.d/mamba.sh
mamba activate ${ENV_NAME}

echo "=== 2. PyTorch のビルド ==="
# PyTorchダウンロード時にプロキシ(download-r2.pytorch.org)に引っかかる場合は
# 以前設定した pip config の trusted-host を事前に確認してください
pip config set global.trusted-host "pypi.org files.pythonhosted.org download.pytorch.org download-r2.pytorch.org pypi.nvidia.com"
make TH_VERSION=2.11.0 CUDA_VERSION=12.8

# echo "=== 3. Flash-Attention 3 のビルド ==="
# cd /data/store5/tkano/flash-attention/hopper
# export TORCH_CUDA_ARCH_LIST="9.0;10.0"  # 閉じクォーテーションを修正
# # 古いキャッシュを無視してクリーンビルド
# PYTHONHTTPSVERIFY=0 MAX_JOBS=16 pip install . --no-build-isolation --no-cache-dir

echo "=== 4. Transformers のビルド ==="
cd /data/store5/tkano/transformers
pip install -e .

# echo "=== 5. warp-transducer のカスタムビルド ==="
# cd /data/store5/tkano/espnet/tools
# ./installers/install_warp-transducer.sh
