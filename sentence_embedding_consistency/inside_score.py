import pandas as pd 
import numpy as np
import torch 
import ast
import fire 


def load_pandas_data(path='/app/data/human_arm_with_embeddings_20'): 
    df = pd.read_csv(path, index_col=0)

    if df['text_embeddings'].dtype == object:
        df['text_embeddings'] = df['text_embeddings'].apply(ast.literal_eval)
    
    return df  

def create_embedding_matrix(dataframe, embedding_column_title='text_embeddings'): 
    return torch.tensor(dataframe[embedding_column_title].tolist()).T

def compute_covariance_matrix(Z):
    d, K = Z.shape
    
    # Compute the centering matrix J_d
    I_d = torch.eye(d)
    ones_d = torch.ones((d, 1))
    J_d = I_d - (1 / d) * (ones_d @ ones_d.T)
    
    # Compute covariance as per Eq. 4 @ https://arxiv.org/abs/2402.03744
    sigma = Z.T @ J_d @ Z
    
    return sigma

def regularise_covariance_matrix(sigma, alpha=0.001):
    K = sigma.size()[0] # number of outputs 
    return sigma + (alpha * torch.eye(K))

def compute_eigen_score(covariance_matrix):
    U, S, V = torch.svd(covariance_matrix)
    
    eigenvalues = S.pow(2)

    K = eigenvalues.size()[0]
    eigen_score = (1/K) * torch.sum(torch.log(eigenvalues))

    return eigen_score 

def main(path: str = '/Users/kqbg611/Documents/data/human_arm_with_embeddings_20'): 
    df = load_pandas_data(path)
    embeddings = create_embedding_matrix(df)
    covariance_matrix = compute_covariance_matrix(embeddings)
    covariance_matrix = regularise_covariance_matrix(covariance_matrix)
    print(compute_eigen_score(covariance_matrix))

    
if __name__ == '__main__':
    fire.Fire(main)

