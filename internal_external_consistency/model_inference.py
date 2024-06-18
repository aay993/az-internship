import fire
import os
import sys
import time

import torch
from transformers import AutoTokenizer
from model_utils import load_model

def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_gpu: bool = False,
    **kwargs
): 
    def inference(user_prompt, temperature, top_p, top_k, max_new_tokens, **kwargs,):
        model = load_model(model_name, quantization, use_fast_kernels, use_gpu)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name)   
        if tokenizer.pad_token_id is None:
            print("Current pad token is None. Adding pad token to tokenizer.")
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            print(f'Previous model vocab size: {model.config.vocab_size}')
            model.resize_token_embeddings(len(tokenizer))
            print(f'Currect model vocab size: {model.config.vocab_size}')

        # Set the pad token id in the model
        model.config.pad_token_id = tokenizer.pad_token_id 
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt") # dict of input ids and attn masks
        if use_gpu:
            batch = {k: v.to('cuda') for k, v in batch.items()}

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                output_hidden_states=True, 
                return_dict_in_generate=True
            )
        e2e_inference_time = (time.perf_counter() - start_time)*1000 
        print(f"End-to-end inference time: {e2e_inference_time:.2f} ms") 

        generated_tokens = outputs.sequences
        decoded_output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        print("Generated Output:", decoded_output)

        hidden_states = outputs.hidden_states

        # grab the embeddings of the final token (represents sentence embedding)
        hidden_states = hidden_states[-1]
        middle_layer_index = (len(hidden_states) + 1) // 2 # +1 to account for input embeddings as per https://huggingface.co/docs/transformers/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput
        middle_layer_hidden_states = hidden_states[middle_layer_index]
        print("Middle layer hidden states:", middle_layer_hidden_states.squeeze().size())
        
    if prompt_file is not None: 
        assert os.path.exists(prompt_file), f"Prompt file {prompt_file} not found."
        with open(prompt_file, 'r') as f:
            user_prompt = "\n".join(f.readlines())
        inference(user_prompt, temperature, top_p, top_k, max_new_tokens)

if __name__ == "__main__":
    fire.Fire(main) 