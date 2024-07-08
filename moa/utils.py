import os
import json
import time
import copy
import uuid
import requests
import openai
import torch
import numpy as np
from openai import OpenAI
from loguru import logger
from dotenv import load_dotenv
# from tenacity import retry, wait_random_exponential, stop_after_attempt
from sentence_embedding_consistency.divergence_inside_score_numpy import compute_divergence

load_dotenv()

client = OpenAI()

DEBUG = int(os.environ.get("DEBUG", "0"))

def generate_together(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    streaming=False,
):

    output = None

    for sleep_time in [1, 2, 4, 8, 16, 32]:

        try:

            endpoint = "https://api.together.xyz/v1/chat/completions"

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model}`."
                )

            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY')}",
                },
            )
            if "error" in res.json():
                logger.error(res.json())
                if res.json()["error"]["type"] == "invalid_request_error":
                    logger.info("Input + output is longer than max_position_id.")
                    return None

            output = res.json()["choices"][0]["message"]["content"]

            break

        except Exception as e:
            logger.error(e)
            if DEBUG:
                logger.debug(f"Msgs: `{messages}`")

            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:

        return output

    output = output.strip()

    if DEBUG:
        logger.debug(f"Output: `{output[:20]}...`.")

    return output


def generate_together_stream(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):
    endpoint = "https://api.together.xyz/v1"
    client = openai.OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY"), base_url=endpoint
    )
    endpoint = "https://api.together.xyz/v1/chat/completions"
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature if temperature > 1e-4 else 0,
        max_tokens=max_tokens,
        stream=True,  # this time, we set stream=True
    )

    return response


def generate_openai(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):

    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()

    return output


def inject_references_to_messages(
    messages,
    references,
):

    messages = copy.deepcopy(messages)

    system = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

    for i, reference in enumerate(references):

        system += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":

        messages[0]["content"] += "\n\n" + system

    else:

        messages = [{"role": "system", "content": system}] + messages
    
    # print(f'the injected message is: {messages}')

    return messages

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def generate_with_references(
    model,
    messages,
    references=[],
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_together,
    reference_models=None
):

    if len(references) > 0:
        try:
            embeddings = [get_embedding(ref) for ref in references]
            logger.debug(f"{len(embeddings)} embeddings fetched")
            embedding_array = np.array(embeddings).T

            divergence_scores = compute_divergence(embedding_array)
            highest_divergence_indices = np.argsort(divergence_scores)[-2:]
            references = [ref for i, ref in enumerate(references) if i not in highest_divergence_indices]

            all_models_info = []
            for i, model_name in enumerate(reference_models):
                model_info = {
                    'model_name': model_name,
                    'filtered_out': i in highest_divergence_indices,
                    'divergence_score': divergence_scores[i]
                }
                all_models_info.append(model_info)

            if not os.path.exists('outputs/filter'):
                os.makedirs('outputs/filter')

            with open(f'outputs/filter/gsm8k_{str(uuid.uuid4())}.json', 'w') as f:
                json.dump(all_models_info, f)
            
        except Exception as e:  
            logger.error(f'Error: {e}')
            return None
        
        messages = inject_references_to_messages(messages, references)
        print('')

    return generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
