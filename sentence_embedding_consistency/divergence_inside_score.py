from sentence_embedding_consistency.inside_score import * 
import fire 

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

def main(path: str = '/Users/kqbg611/Documents/data/human_arm_with_embeddings_20'):
    df = load_pandas_data(path) 
    embeddings = create_embedding_matrix(df)
    print(embeddings.size())
    divs = compute_divergence(embeddings)
    sorted_indices = sorted(range(len(divs)), key=lambda i: divs[i], reverse=True)
    top_5_indices = sorted_indices[:5]
    for i in top_5_indices:
        print(f"Embedding {i}: Divergence = {divs[i]}")
        print(df['text_responses'][i])
    # Now let's print the least 5 divergent embeddings
    bottom_5_indices = sorted_indices[-5:]
    for i in bottom_5_indices:
        print(f"Embedding {i}: Divergence = {divs[i]}")
        print(df['text_responses'][i])

if __name__ == '__main__':
    fire.Fire(main)