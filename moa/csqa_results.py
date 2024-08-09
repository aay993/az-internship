import json
import fire

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_answer(output):
    # This function assumes the answer is always in parentheses 
    import re
    match = re.search(r'\(([a-e])\)', output.lower())
    if match:
        return match.group(1).upper()
    return None

def calculate_accuracy(data):
    correct = 0
    total = len(data)
    
    for item in data:
        extracted_answer = extract_answer(item['output'])
        if extracted_answer == item['answerKey']:
            correct += 1 
        else:
            print(f'the question is {item["question"]}')
    
    return correct / total if total > 0 else 0

def main(json_path):
    data = load_json(json_path)
    accuracy = calculate_accuracy(data)
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    fire.Fire(main)