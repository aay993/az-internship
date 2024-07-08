import json
import re
import fire

def check_numerical_answer(output, numerical_answer):
    if output is None:
        return False
    output_numeric = re.sub(r'[^0-9]', '', output)
    return numerical_answer in output_numeric

def calculate_accuracy(data):
    correct_count = 0
    none_count = 0

    for item in data:
        numerical_answer = item['numerical_answer']
        output = item['output']
        
        if output is None:
            none_count += 1
            print(f"Debug: output is None")
        elif check_numerical_answer(output, numerical_answer):
            correct_count += 1
        else:
            print(f"Debug: Incorrect answer")

    total_count = len(data)
    accuracy = correct_count / total_count * 100
    return correct_count, total_count, accuracy, none_count

def main(path:str): 
    file_path = path 

    with open(file_path, 'r') as file:
        data = json.load(file)
    
    correct_count, total_count, accuracy, none_count = calculate_accuracy(data)
    print(f"Number of correct answers: {correct_count}")
    print(f"Total number of items: {total_count}")
    print(f"Number of None outputs: {none_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    
if __name__ == "__main__":
    fire.Fire(main)