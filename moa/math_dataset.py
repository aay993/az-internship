import random
from datasets import load_dataset
from math_eval_utils_2 import extract_math_answer, eval_math

INVALID_ANS = "Invalid"

def load_math_dataset():
    """
    Load the hendrycks/competition_math dataset using the datasets package.

    Returns:
        dataset (DatasetDict): The loaded hendrycks/competition_math dataset.
    """
    dataset = load_dataset("hendrycks/competition_math")
    return dataset

def add_extracted_answer_column(example, extract_predicted_answer=False):
    if extract_predicted_answer:
        example['predicted_answer'] = extract_math_answer(example['problem'], example['output'], "solve")
        if example['predicted_answer'] is None:
            example['predicted_answer'] = INVALID_ANS
    else:
        example['extracted_answer'] = extract_math_answer(example['problem'], example['solution'], "solve")
        if example['extracted_answer'] is None:
            example['extracted_answer'] = INVALID_ANS
    return example

def generate_cot_prompt(dataset, n=5):
    """
    Generate a 5-shot CoT prompt from the training dataset.

    Args:
        dataset (DatasetDict): The MATH dataset.
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
        cot_prompt += f"Problem: {example['problem']}\n"
        cot_prompt += f"Solution: {example['solution']}\n\n"
    
    return cot_prompt


def process_math_dataset(num_samples=5):
    dataset = load_dataset("hendrycks/competition_math")
    
    for i in range(num_samples):
        sample = dataset['train'][i]
        problem = sample['problem']
        actual_solution = sample['solution']
        
        # Simulate LLM prediction (replace this with actual LLM call in practice)
        predicted_solution = "To ensure that the function f(x) is continuous, the pieces of the function must connect seamlessly at the boundaries, namely at x = -2 and x = 2. First, we check the boundary at x = 2. The function definition from the right of x = 2 is given by f(x) = ax + 3, and from the left, f(x) = x - 5. For continuity at x = 2: lim_{x to 2^+} f(x) = lim_{x to 2^-} f(x) Calculating each side, we find: lim_{x to 2^+} f(x) = 2a + 3 lim_{x to 2^-} f(x) = 2 - 5 = -3 Equating the two limits, we have: 2a + 3 = -3 Solving for a, we get: 2a = -6 => a = -3 Next, we check the boundary at x = -2. The function definition from the right of x = -2 is given by f(x) = x - 5, and from the left, f(x) = 2x - b. For continuity at x = -2: lim_{x to -2^+} f(x) = lim_{x to -2^-} f(x) Calculating each side, we find: lim_{x to -2^+} f(x) = -2 - 5 = -7 lim_{x to -2^-} f(x) = -4 - b Equating the two limits, we have: -4 - b = -7 Solving for b, we get: -b = -3 => b = 3 Thus, the values of a and b are -3 and 3 respectively. The sum a + b is: a + b = -3 + 3 = 0 Therefore, a + b = 0."  # This is just a placeholder
        
        # Extract answer from the predicted solution
        extracted_answer = extract_math_answer(problem, predicted_solution, "solve")
        
        # Evaluate the extracted answer
        evaluation_result = eval_math({
            'prediction': extracted_answer,
            'answer': extract_math_answer(problem, actual_solution, "solve")
        })
        
        print(f"Sample {i+1}:")
        print(f"Problem: {problem[:100]}...")  # Print first 100 characters of the problem
        print(f"Extracted Answer: {extracted_answer}")
        print(f"Actual Solution: {actual_solution[:100]}...")  # Print first 100 characters of the actual solution
        print(f"Evaluation Result: {'Correct' if evaluation_result else 'Incorrect'}")
        print("\n")

if __name__ == "__main__":
    math_dataset = load_math_dataset()
    math_dataset = math_dataset.map(add_extracted_answer_column)
    
    assert all(example['extracted_answer'] != INVALID_ANS for example in math_dataset['train']), "Invalid answer found in train dataset"
    assert all(example['extracted_answer'] != INVALID_ANS for example in math_dataset['test']), "Invalid answer found in test dataset"

    cot_prompt = generate_cot_prompt(math_dataset, n=4)
    print("Generated 5-shot CoT Prompt:")
    print(cot_prompt)