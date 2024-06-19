from typing import Tuple
import pandas as pd 
import torch 
import torch.nn.functional as F 
from inside_score import load_pandas_data 

def create_ball_embeddings(dataframe: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    unperturbed_embedding = torch.tensor(dataframe['text_embeddings'][0])
    ball_embeddings = torch.tensor(dataframe['text_embeddings'][1:].tolist())
    ball_embeddings_avg = torch.mean(ball_embeddings, dim=0)
    
    return unperturbed_embedding, ball_embeddings_avg 

def harmonic_score(output_vector: torch.Tensor, ball_vector: torch.Tensor):
    if not torch.isclose(output_vector.norm(p=2), torch.tensor(1.0)):
        print('output_vector not actually normalised to 1; normalising now')
        output_vector = F.normalize(output_vector, p=2, dim=0)
    if not torch.isclose(ball_vector.norm(p=2), torch.tensor(1.0)):
        print('ball_vector not actually normalised to 1; normalising now')
        ball_vector = F.normalize(ball_vector, p=2, dim=0) 
    
    cosine_sim = F.cosine_similarity(output_vector, ball_vector, dim=0)
    
    return cosine_sim.item()

if __name__ == '__main__': 
    df = load_pandas_data('/app/data/human_arm_with_embeddings_20')
    orig, avg = create_ball_embeddings(df)
    score = harmonic_score(orig, avg)
    print(f"Harmonic score (cosine similarity): {score}")
    print(orig.size(), avg.size())
