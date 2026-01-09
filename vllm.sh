export CUDA_VISIBLE_DEVICES=0
export TP=1
export MODEL_PATH=/root/autodl-tmp/qwen3_moe_small
export MODEL_NAME=Qwen3-14B
export PORT=10133
export ARCTIC_INFERENCE_ENABLED=1
export VLLM_USE_CUDA_GRAPH=0

python3 -m vllm.entrypoints.openai.api_server  --host 0.0.0.0 --port ${PORT} --dtype bfloat16 --model ${MODEL_PATH} --served-model-name ${MODEL_NAME} --tensor-parallel-size ${TP} --gpu-memory-utilization 0.9 --enforce-eager   --max-model-len  32768 --trust-remote-code --no-enable-prefix-caching --speculative_config '{"method": "arctic",
        "model":"/root/output/global_step_2/speculator",
        "num_speculative_tokens": 3,
        "enable_suffix_decoding": true}'

