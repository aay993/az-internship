import json 
import torch 

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f) 

def process_embeddings(data):
    for item in data:
        if 'sentence_embedding_matrix' in item:
            # Convert the list back to a torch tensor
            item['sentence_embedding_matrix'] = torch.tensor(item['sentence_embedding_matrix'])
    return data

if __name__ == '__main__':
    file_path = '/home/aay993/astrazeneca/az-internship/internal_external_consistency/alpaca_results.json'
    data = load_json(file_path)
    processed_data = process_embeddings(data)
    print(processed_data[:1])
    