#!/bin/bash
# SorGPT API Server 启动脚本
# 修复: LD_PRELOAD conda libstdc++ (CXXABI fix) + MKL sequential (避免死锁)

cd /vol/sunjilin/website/data/agent/sorghum_rag

export LD_PRELOAD=/home/sunjilin/miniforge3/envs/sorghum_rag/lib/libstdc++.so.6
export MKL_THREADING_LAYER=sequential
export OMP_NUM_THREADS=1

# 加载 .env 密钥（密钥统一从环境变量注入，避免硬编码进源码）
set -a; source .env; set +a

source /home/sunjilin/miniforge3/etc/profile.d/conda.sh
conda activate sorghum_rag

nohup python api_server.py > app.log 2>&1 &
echo "pid=$!"
