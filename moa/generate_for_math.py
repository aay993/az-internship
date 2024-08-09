import json
import os
import datasets
from dotenv import load_dotenv
from fire import Fire
from functools import partial
from typing import List
from loguru import logger
from math_dataset import load_math_dataset, generate_cot_prompt, add_extracted_answer_column
from utils import (
    generate_together,
    generate_openai,
    generate_with_references,
    DEBUG,
)

load_dotenv()

def process_fn(
    item,
    model,
    train_data,  
    reference_models=[],
    temperature=0.7,
    max_tokens=2048,
    rounds=1,
):
    
    cot_prompt = generate_cot_prompt(train_data, n=4)
    
    messages = [{"role": "user", "content": cot_prompt + '\nProblem: ' + item["problem"]}]

    references = item.get("references", [])

    if len(references) == 0 and len(reference_models) > 0:

        prev_references = []

        for i_round in range(rounds):

            if DEBUG:
                logger.info(
                    f"Round {i_round+1}/{rounds} to collecting reference responses."
                )

            references = []

            for reference_model in reference_models:

                reference = generate_with_references(
                    model=reference_model,
                    messages=messages,
                    references=prev_references,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reference_models=reference_models,
                    save_model_usage=True, 
                    save_model_usage_path='outputs/math'
                )

                if reference is not None:

                    references.append(reference)

            if i_round < rounds - 1:

                prev_references = references

                references = []

    output = generate_with_references(
        model=model,
        messages=messages,
        references=references,
        reference_models=reference_models,
        save_model_usage=True, 
        save_model_usage_path='outputs/math'
    )

    return {"output": output, "generator": model + "-together"}

def main(
    model: str,
    output_path: str,
    reference_paths: str = None,
    reference_models: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    rounds: int = 1,
    num_proc: int = 16,
):

    if reference_paths is None:
        reference_paths = []
    else:
        reference_paths = reference_paths.split(",")

    if reference_models is None:
        reference_models = []
    else:
        reference_models = reference_models.split(",")

    math_dataset = load_math_dataset()
    data = math_dataset # we pass the data to process_fn, subselect the training data and generate a 5-shot prompt
    eval_set = math_dataset['test']
    eval_set = eval_set.select(range(1500)) # sample for testing
    eval_set = eval_set.map(add_extracted_answer_column)

    if len(reference_paths):

        logger.info(f"`reference_paths` provided: {reference_paths}")

        references = []
        for reference_path in reference_paths:
            with open(reference_path) as f:
                reference_responses = json.load(f)
                logger.info(
                    f"Reading reference outputs: {reference_path} ({len(reference_responses)})"
                )
                for i_reference_response, reference_response in enumerate(
                    reference_responses
                ):
                    if len(references) <= i_reference_response:
                        references.append([reference_response["output"]])
                    else:
                        references[i_reference_response].append(
                            reference_response["output"]
                        )

        eval_set = eval_set.add_column(f"references", references)

    elif len(reference_models):

        logger.info(
            f"`reference_models` provided: {reference_models}. Will generate reference responses on-the-fly."
        )

    logger.info(f"Start.")

    eval_set = eval_set.map(
        partial(
            process_fn,
            model=model,
            train_data=data,  
            reference_models=reference_models,
            temperature=temperature,
            max_tokens=max_tokens,
            rounds=rounds,
        ),
        batched=False,
        num_proc=num_proc,
    )

    logger.info(f"Saving outputs to {output_path}.")

    try:
        eval_set = eval_set.remove_columns(f"references")
    except Exception as e:
        pass

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, "w") as f:
        json.dump(list(eval_set), f, indent=2)


if __name__ == "__main__":

    Fire(main)