import re
import random
from datasets import load_dataset

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def download_gsm8k():
    """
    Download the GSM8K dataset using the datasets package.

    Returns:
        dataset (DatasetDict): The loaded GSM8K dataset.
    """
    dataset = load_dataset("gsm8k", 'main')
    return dataset

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def add_numerical_column(example):
    example['numerical_answer'] = extract_answer(example['answer'])
    return example

def generate_cot_prompt(dataset, n=5):
    """
    Generate a 5-shot CoT prompt from the training dataset.

    Args:
        dataset (DatasetDict): The GSM8K dataset.
        n (int): The number of examples to include in the prompt.

    Returns:
        str: The generated CoT prompt.
    """
    # Select n random examples from the training split
    train_examples = random.sample(list(dataset['train']), n)
    
    # Format the prompt
    cot_prompt = ""
    for i, example in enumerate(train_examples, 1):
        cot_prompt += f"Example {i}:\n"
        cot_prompt += f"Question: {example['question']}\n"
        cot_prompt += f"Answer: {example['answer']}\n\n"
    
    return cot_prompt

# Example usage
if __name__ == "__main__":
    gsm8k_dataset = download_gsm8k()
    gsm8k_dataset = gsm8k_dataset.map(add_numerical_column)
    print(gsm8k_dataset['test'])

    assert all(example['numerical_answer'] != INVALID_ANS for example in gsm8k_dataset['train']), "Invalid answer found in train dataset"
    assert all(example['numerical_answer'] != INVALID_ANS for example in gsm8k_dataset['test']), "Invalid answer found in test dataset"

    # Generate a 5-shot CoT prompt
    cot_prompt = generate_cot_prompt(gsm8k_dataset, n=5)
    print("Generated 5-shot CoT Prompt:")
    print(cot_prompt)