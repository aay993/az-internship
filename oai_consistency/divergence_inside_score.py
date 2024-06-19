from inside_score import * 

ALPHA = 0.001

def compute_divergence(embeddings, alpha=ALPHA): 
    overall_covariance_matrix = compute_covariance_matrix(embeddings)
    regularised_overall_covariance_matrix = regularise_covariance_matrix(overall_covariance_matrix, alpha) 
    overall_eigenscore = compute_eigen_score(regularised_overall_covariance_matrix)

    divergences = []

    K = embeddings.shape[1] # columns relate to K number of outputs for that particular input 

    for i in range(K): 
        reduced_embeddings = torch.cat((embeddings[:,:i], embeddings[:, i+1:]), dim=1) 
        
        reduced_covariance_matrix = compute_covariance_matrix(reduced_embeddings)
        regularised_reduced_covariance_matrix = regularise_covariance_matrix(reduced_covariance_matrix, alpha)
        reduced_eigenscore = compute_eigen_score(regularised_reduced_covariance_matrix)

        divergence = overall_eigenscore - reduced_eigenscore
        divergences.append(divergence.item())
    
    return divergences

if __name__ == '__main__':
    df = load_pandas_data() 
    print(df)
    embeddings = create_embedding_matrix(df)
    divs = compute_divergence(embeddings)
    for i, divergence in enumerate(divs):
        print(f"Embedding {i}: Divergence = {divergence}")