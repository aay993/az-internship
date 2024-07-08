import pandas as pd
import numpy as np
import ast
import fire

def load_pandas_data(path='/app/data/human_arm_with_embeddings_20'):
    df = pd.read_csv(path, index_col=0)

    if df['text_embeddings'].dtype == object:
        df['text_embeddings'] = df['text_embeddings'].apply(ast.literal_eval)

    return df

def create_embedding_matrix(dataframe, embedding_column_title='text_embeddings'):
    return np.array(dataframe[embedding_column_title].tolist()).T

def compute_covariance_matrix(Z):
    d, K = Z.shape

    # Compute the centering matrix J_d
    I_d = np.eye(d)
    ones_d = np.ones((d, 1))
    J_d = I_d - (1 / d) * (ones_d @ ones_d.T)

    # Compute covariance
    sigma = Z.T @ J_d @ Z

    return sigma

def regularise_covariance_matrix(sigma, alpha=0.001):
    K = sigma.shape[0]  # number of outputs
    return sigma + (alpha * np.eye(K))

def compute_eigen_score(covariance_matrix):
    U, S, V = np.linalg.svd(covariance_matrix)

    eigenvalues = S**2

    K = len(eigenvalues)
    eigen_score = (1/K) * np.sum(np.log(eigenvalues))

    return eigen_score

def main(path: str = '/Users/kqbg611/Documents/data/human_arm_with_embeddings_20'):
    df = load_pandas_data(path)
    embeddings = create_embedding_matrix(df)
    covariance_matrix = compute_covariance_matrix(embeddings)
    covariance_matrix = regularise_covariance_matrix(covariance_matrix)
    print(compute_eigen_score(covariance_matrix))

if __name__ == '__main__':
    fire.Fire(main)