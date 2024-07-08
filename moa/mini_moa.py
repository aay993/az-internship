import asyncio
import os
import copy
from typing import List

import torch 
import openai
from together import AsyncTogether, Together
from dotenv import load_dotenv

from sentence_embedding_consistency.divergence_inside_score import compute_divergence
from gsm8k_dataset import download_gsm8k, generate_cot_prompt, add_numerical_column

load_dotenv()

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
client_oai = openai.AsyncOpenAI() 

gsm8k_dataset = download_gsm8k()
gsm8k_dataset = gsm8k_dataset.map(add_numerical_column)
cot_prompt = generate_cot_prompt(gsm8k_dataset, n=5)

user_prompt = {"role": "user", "content": cot_prompt + '\n' + gsm8k_dataset['test'][73]['question']}

reference_models = [
    "microsoft/WizardLM-2-8x22B",
    "Qwen/Qwen1.5-110B-Chat",
    "Qwen/Qwen1.5-72B-Chat",
    "meta-llama/Llama-3-70b-chat-hf",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "databricks/dbrx-instruct"
]

aggregator_model = "Qwen/Qwen1.5-110B-Chat"
aggreagator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

def inject_references_to_messages(messages, references):
    messages = copy.deepcopy(messages)
    system = aggreagator_system_prompt
    for i, reference in enumerate(references):
        system += f"\n{i+1}. {reference}"
    if messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + system
    else:
        messages = [{"role": "system", "content": system}] + messages
    return messages

async def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    text = text.replace("\n", " ")
    response = await client_oai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

async def embed_responses(responses: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    tasks = [get_embedding(text, model) for text in responses]
    embeddings = await asyncio.gather(*tasks)
    return embeddings

async def run_llm(model):
    response = await async_client.chat.completions.create(
        model=model,
        messages=[user_prompt],
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content

async def main():
    # Step 1: Retrieve the results from the LLMs
    results = await asyncio.gather(*[run_llm(model) for model in reference_models])

    random_text = "Ay Yo Breezy what you saying breh? How's the weather today?"
    results.append(random_text)
    reference_models.append("random_text")

    # Step 2: Embed each result using the TAI model
    embeddings = await embed_responses(results)

    # Step 3: Combine all the embeddings into a K x D torch.tensor matrix
    embedding_tensor = torch.tensor(embeddings).T

    divergence_scores = compute_divergence(embedding_tensor)

    # Create a list of tuples (index, divergence_score, model)
    indexed_scores = [(index, score, reference_models[index]) for index, score in enumerate(divergence_scores)]
    # Sort the list based on divergence scores
    indexed_scores.sort(key=lambda x: x[1])

    # Get the top 3 results with highest divergence scores
    top_3_highest = indexed_scores[-3:]
    # Get the top 2 results with lowest divergence scores
    top_2_lowest = indexed_scores[:2]

    print("Top 3 results with highest divergence scores:")
    for index, score, model in top_3_highest:
        print(f"Index: {index}, Score: {score}, Model: {model}, Response: {results[index]}")

    print("\nTop 2 results with lowest divergence scores:")
    for index, score, model in top_2_lowest:
        print(f"Index: {index}, Score: {score}, Model: {model}, Response: {results[index]}")
    
    filtered_results = [results[index] for index, _, _ in indexed_scores[:-3]]

    # Inject the references to messages for the aggregator model
    final_messages = inject_references_to_messages([user_prompt], filtered_results)

    # Get the final aggregated response
    finalStream = client.chat.completions.create(
        model=aggregator_model,
        messages=final_messages,
        stream=True,
    )

    for chunk in finalStream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
    print()
    print('done')
    print(gsm8k_dataset['test'][73])