from datasets import load_dataset 

def download_csqa():
    """
    Download the CSQA dataset using the datasets package.

    Returns:
        dataset (DatasetDict): The loaded CSQA dataset.
    """
    dataset = load_dataset("commonsense_qa")
    return dataset 

def format_choices(choices):
    """
    Format the choices for a CSQA example.

    Args:
        choices dict: The choices for a CSQA example.

    Returns:
        str: The formatted choices.
    """
    labels = choices['label']
    texts = choices['text'] 

    formatted_choices = []
    for label, text in zip(labels, texts):
        formatted_choices.append(f"({label.lower()}) {text}")
    return '\n'.join(formatted_choices)

def add_formated_choice_column(example):
    example['formatted_choices'] = format_choices(example['choices'])
    return example

def return_wang_cot_str():
    return "Q: Sammy wanted to go to where the people were. Where might he go?\nAnswer Choices:\n(a) populated areas\n(b) race track\n(c) desert\n(d) apartment\n(e) roadblock\nA: The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. The answer is (a).\n\nQ: The fox walked from the city into the forest, what was it looking for?\nAnswer Choices:\n(a) pretty flowers\n(b) hen house\n(c) natural habitat\n(d) storybook\nA: Answer: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. The answer is (b).\n\nQ: What home entertainment equipment requires cable?\nAnswer Choices:\n(a) radio shack\n(b) substation\n(c) television\n(d) cabinet\nA: The answer must require cable. Of the above choices, only television requires cable. The answer is (c).\n\nQ: Google Maps and other highway and street GPS services have replaced what?\nAnswer Choices:\n(a) united states\n(b) mexico\n(c) countryside\n(d) atlas\nA: The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. The answer is (d).\n\nQ: What do people use to absorb extra ink from a fountain pen?\nAnswer Choices:\n(a) shirt pocket\n(b) calligrapher's hand\n(c) inkwell\n(d) desk drawer\n(e) blotter\nA: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. The answer is (e).\n\nQ: Where do you put your grapes just before checking out?\nAnswer Choices:\n(a) mouth\n(b) grocery cart\n(c)super market\n(d) fruit basket\n(e) fruit market\nA: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. The answer is (b).\n\nQ: Before getting a divorce, what did the wife feel who was doing all the work?\nAnswer Choices:\n(a) harder\n(b) anguish\n(c) bitterness\n(d) tears\n(e) sadness\nA: The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. The answer is (c).\n\n"

def generate_user_query(item, CoT=None): 
    if CoT is None: 
        CoT = return_wang_cot_str()
    
    question = item['question']
    choices = item['formatted_choices']
    return CoT + f"Q: {question}\nAnswer Choices:\n{choices}\n" + "\nA:"

if __name__ == '__main__': 
    csqa_dataset = download_csqa() 
    csqa_dataset = csqa_dataset.map(add_formated_choice_column)
    cot = return_wang_cot_str()
    print(generate_user_query(csqa_dataset['train'][8], cot)) 
   
