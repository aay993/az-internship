export DEBUG=1

# reference_models="microsoft/WizardLM-2-8x22B,Qwen/Qwen1.5-110B-Chat,Qwen/Qwen1.5-72B-Chat,meta-llama/Llama-3-70b-chat-hf,mistralai/Mixtral-8x22B-Instruct-v0.1,databricks/dbrx-instruct"

reference_models="meta-llama/Llama-3-70b-chat-hf,meta-llama/Llama-3-70b-chat-hf,meta-llama/Llama-3-70b-chat-hf,meta-llama/Llama-3-70b-chat-hf,meta-llama/Llama-3-70b-chat-hf,meta-llama/Llama-3-70b-chat-hf"

# reference_models="meta-llama/Llama-3-70b-chat-hf,meta-llama/Llama-3-70b-chat-hf,meta-llama/Llama-3-70b-chat-hf"

python generate_for_gsm8k.py \
    --model="Qwen/Qwen1.5-72B-Chat" \
    --output-path="outputs/gsm8k-Qwen-72B-round-1_MoA-Lite-f-4.json" \
    --reference-models=${reference_models} \
    --rounds 1 \
    --num-proc 10

python gsm8k_results.py \
    --path="outputs/gsm8k-Qwen-72B-round-1_MoA-Lite-f-4.json" 