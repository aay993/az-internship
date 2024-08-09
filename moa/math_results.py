import json
import fire
from datasets import Dataset
from math_eval_utils_2 import eval_math
from math_dataset import add_extracted_answer_column, INVALID_ANS

def main(path: str):
    # Load the JSON file
    with open(path, 'r') as file:
        data = json.load(file)
    
    # Convert the JSON data to a Hugging Face dataset
    dataset = Dataset.from_list(data)
    
    # Extract predicted answers using the map function
    dataset = dataset.map(lambda example: add_extracted_answer_column(example, extract_predicted_answer=True))
    
    # Extract ground truth answers (if not already present)
    if 'extracted_answer' not in dataset.column_names:
        dataset = dataset.map(lambda example: add_extracted_answer_column(example, extract_predicted_answer=False))
    
    # # Print a sample to verify
    # print(dataset[0])
    
    # Evaluate the predictions
    def evaluate_prediction(example):
        is_correct = eval_math({
            'prediction': example['predicted_answer'],
            'answer': example['extracted_answer']
        })
        return {"is_correct": is_correct}
    
    dataset = dataset.map(evaluate_prediction)
    
    # Print statistics
    correct_count = sum(dataset['is_correct'])
    total_count = len(dataset)
    accuracy = correct_count / total_count
    
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    
    # Count INVALID_ANS occurrences
    invalid_predicted = sum(1 for ans in dataset['predicted_answer'] if ans == INVALID_ANS)
    invalid_extracted = sum(1 for ans in dataset['extracted_answer'] if ans == INVALID_ANS)
    
    print(f"Invalid predicted answers: {invalid_predicted}/{total_count}")
    print(f"Invalid extracted answers: {invalid_extracted}/{total_count}")
    
    # Save the dataset as a JSON file
    with open('outputs/processed_math_dataset.json', "w") as f:
        json.dump(list(dataset), f, indent=2)


if __name__ == "__main__":
    fire.Fire(main)