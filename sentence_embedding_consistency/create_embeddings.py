import logging
import os 
import random
from typing import List

import pandas as pd
import asyncio
import openai
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
client = openai.AsyncOpenAI()

def perturb_query(query: str, num_chars: int = 3, random_insert: bool = False) -> str:
    """Inserts up to `num_chars` random ASCII control characters into the query.
    
    Args:
        query (str): The original query string.
        num_chars (int): Maximum number of ASCII control characters to add.
        random_insert (bool): If True, insert characters at random positions.
                               If False, append characters at the end of the query.

    Returns:
        str: The perturbed query.
    """
    chars_to_insert = [chr(random.randint(0, 31)) for _ in range(random.randint(1, num_chars))]
    
    if random_insert:
        for char in chars_to_insert:
            pos = random.randint(0, len(query))
            query = query[:pos] + char + query[pos:]
    else:
        query += ''.join(chars_to_insert)
    
    return query

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def fetch_response(query: str) -> str: 
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

async def create_responses(
    query: str = 'Which bones are located in the human arm?', 
    num_responses: int = 20,
    perturb: bool = False) -> pd.DataFrame:

    # Fetch the first response separately
    first_response_future = fetch_response(query)
    first_response = await first_response_future
    responses = [first_response]

    tasks = []
    for i in range(1, num_responses):
        if perturb:
            perturbed_query = perturb_query(query, num_chars=3)
            tasks.append(fetch_response(perturbed_query))
        else:
            tasks.append(fetch_response(query))

    for i, task in enumerate(asyncio.as_completed(tasks)):
        try:
            response = await task
            responses.append(response)
            logger.info(f"Fetched response {i + 2}/{num_responses}")  # Adjust index for logging
        except Exception as e:
            logger.error(f"Failed to fetch response {i + 2}/{num_responses}: {e}")
            responses.append(None)

    # Create DataFrame from responses
    df = pd.DataFrame(responses, columns=['text_responses'])
    return df

async def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    text = text.replace("\n", " ")
    response = await client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

async def embed_responses(df: pd.DataFrame) -> pd.DataFrame:
    tasks = [get_embedding(text) for text in df['text_responses']]
    embeddings = await asyncio.gather(*tasks)
    df['text_embeddings'] = embeddings
    return df

async def main(): 
    df_responses = await create_responses(
        query='Q: Bob wants to warm his hands by rubbing them. Which skin surface will produce most heat? (A) dry palms. (B) wet palms A: Dry surfaces will more likely cause more friction via rubbing. So the answer is (A). Q: Cells need nutrients for energy. Which system breaks down food to cellular energy? (A) digestive (B) excretory (C) circulatory (D) respiratory', 
        num_responses=20, 
        perturb=False)

    new_row = pd.DataFrame({'text_responses': ['The system which breaks down food is the respiratory system, where the lungs break down the food.']})
    df_responses = pd.concat([df_responses, new_row], ignore_index=True)

    df_responses = await embed_responses(df_responses)
    print(df_responses)
    os.makedirs('data', exist_ok=True)
    df_responses.to_csv('data/human_arm_with_embeddings_20')
    
if __name__ == "__main__":
    asyncio.run(main())
