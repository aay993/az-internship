from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig

def load_model(model_name, quantization=False, use_fast_kernels=None, use_gpu=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        return_dict=True,
        load_in_8bit=quantization,
        low_cpu_mem_usage=True, 
        attn_implementation='sdpa' if use_fast_kernels else None,
        device_map='auto' if use_gpu else None
    )
    return model 

# Loading the model from config to load FSDP checkpoints into that
def load_llama_from_config(config_path):
    model_config = LlamaConfig.from_pretrained(config_path) 
    model = LlamaForCausalLM(config=model_config)
    return model

if __name__ == '__main__':
    model_path = '/home/aay993/astrazeneca/models/llama3'
    try:
        model = load_model(model_path)
        print("Model loaded successfully")
        print(model)
    except Exception as e:
        print(f"Model loading failed: {e}")
    