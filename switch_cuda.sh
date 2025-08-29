#!/bin/bash
# 用法：source ./switch_cuda.sh 12.8

set -e
ver="$1"
if [ "$ver" = "12.8" ]; then
  CUDA_HOME=/home1/zhouhao/cuda_12_8
else
  echo "未知版本：$ver（目前支持：12.8）"; return 1
fi

if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "未找到 nvcc: $CUDA_HOME/bin/nvcc"; return 1
fi

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "[OK] 切到 CUDA $ver -> $CUDA_HOME"
which nvcc
nvcc -V
