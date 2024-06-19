import datasets

def load_alpaca_dataset():
    eval_set = datasets.load_dataset(
            "tatsu-lab/alpaca_eval", "alpaca_eval_gpt4_baseline", trust_remote_code=True
        )["eval"]

    eval_set = eval_set.remove_columns(['output', 'generator'])

    return eval_set[:100] 

if __name__ == '__main__':
    alpaca_dataset = load_alpaca_dataset()
    print(alpaca_dataset[:3])