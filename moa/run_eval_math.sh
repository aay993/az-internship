export DEBUG=1

reference_models="meta-llama/Llama-3-70b-chat-hf,meta-llama/Llama-3-70b-chat-hf,meta-llama/Llama-3-70b-chat-hf,meta-llama/Llama-3-70b-chat-hf,meta-llama/Llama-3-70b-chat-hf,meta-llama/Llama-3-70b-chat-hf" 

python generate_for_math.py \
    --model="Qwen/Qwen1.5-72B-Chat" \
    --output-path="outputs/math-Qwen-72B-mix-filtered4.json" \
    --reference-models=${reference_models} \
    --rounds 1 \
    --num-proc 11

python math_results.py --path="/Users/kqbg611/Documents/astrazeneca/az-internship/moa/outputs/math-Qwen-72B-mix-filtered4.json"