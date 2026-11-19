#!/bin/bash

# Deploy the external multimodal LLM (target MLLM) as an OpenAI-compatible server.
# After it starts, set BASE_URL / MODEL in mmdynopt_agent/utils/tools/mm_LM_env.py:
#   BASE_URL = "http://<SERVER_IP>:8006/v1"
#   MODEL    = "Qwen3-VL-32B-Instruct"

export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH=/path/to/your/Qwen3-VL-32B-Instruct
SERVED_MODEL_NAME=Qwen3-VL-32B-Instruct
PORT=8006

python -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_PATH} \
  --port ${PORT} \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --enable-prefix-caching \
  --served-model-name ${SERVED_MODEL_NAME} \
  --max-num-seqs 128
