import asyncio
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from utils import load_json, process_embeddings
from oai_consistency.inside_score import compute_covariance_matrix, regularise_covariance_matrix, compute_eigen_score
from oai_consistency.create_embeddings import get_embedding

async def get_embeddings_for_outputs(outputs):
    tasks = [get_embedding(output) for output in outputs]
    embeddings = await asyncio.gather(*tasks)
    return torch.tensor(embeddings)

async def main(file_path, plot_path): 
    path = file_path
    data = load_json(path)
    data = process_embeddings(data) # turn embeddings into torch tensors 

    internal_scores = []
    external_scores = []

    for item in data:
        if 'sentence_embedding_matrix' in item and 'outputs' in item:
            # internal embedding consistency
            cov = compute_covariance_matrix(item['sentence_embedding_matrix'].T)
            reg_cov = regularise_covariance_matrix(cov)
            eigen_score = compute_eigen_score(reg_cov)
            item['internal_eigenscore'] = eigen_score
            internal_scores.append(eigen_score)
            
            # external embedding consistency
            ext_embeddings = await get_embeddings_for_outputs(item['outputs'])
            ext_cov = compute_covariance_matrix(ext_embeddings.T)
            ext_reg_cov = regularise_covariance_matrix(ext_cov)
            ext_eigen_score = compute_eigen_score(ext_reg_cov)
            item['external_eigenscore'] = ext_eigen_score
            external_scores.append(ext_eigen_score)

    correlation, _ = pearsonr(internal_scores, external_scores)
    print(f"Correlation between internal and external eigenscores: {correlation}")

    plt.figure(figsize=(10, 6))
    plt.scatter(internal_scores, external_scores, alpha=0.6, label="Data Points")
    sns.regplot(x=internal_scores, y=external_scores, scatter=False, color='red', label="Regression Line")
    plt.title('Correlation between Internal and External Eigenscores')
    plt.xlabel('Internal Eigenscore')
    plt.ylabel('External Eigenscore')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    plt.show()
    
if __name__ == '__main__':
    path = '/home/aay993/astrazeneca/az-internship/internal_external_consistency/alpaca_results.json'
    plot_path = 'correlation_plot.pdf'
    asyncio.run(main(path, plot_path))
