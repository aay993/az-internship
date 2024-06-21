import fire
import torch
import matplotlib.pyplot as plt
import numpy as np

def discrete_semantic_entropy(cluster_sizes):
    """
    Calculate the discrete semantic entropy for given cluster sizes.
    
    Parameters:
    cluster_sizes (list of int): List representing the sizes of each cluster.
    
    Returns:
    float: The calculated discrete semantic entropy.
    """
    cluster_sizes = torch.tensor(cluster_sizes, dtype=torch.float32)
    total_members = torch.sum(cluster_sizes)
    probabilities = cluster_sizes / total_members
    se = -torch.sum(probabilities * torch.log(probabilities))
    return se.item()

def generate_se_plots(output_file='semantic_entropy_plot.png'):
    num_clusters_list = [2, 3, 4]
    uniform_distributions = {
        2: [[10, 10], [5, 15]],
        3: [[10, 10, 10], [5, 10, 15], [1, 14, 15]],
        4: [[10, 10, 10, 10], [2, 4, 8, 16]]
    }
    skewed_distributions = {
        2: [[1, 19], [2, 18], [5, 15]],
        3: [[1, 1, 18], [1, 3, 16], [2, 3, 15]],
        4: [[1, 1, 1, 17], [1, 2, 2, 15]]
    }

    fig, axs = plt.subplots(2, len(num_clusters_list), figsize=(15, 10), sharey=True)
    fig.suptitle('Semantic Entropy (SE) Analysis')

    for idx, num_clusters in enumerate(num_clusters_list):
        # Plot for uniform distribution
        se_values = [discrete_semantic_entropy(dist) for dist in uniform_distributions[num_clusters]]
        axs[0, idx].bar(range(len(se_values)), se_values, color='b')
        axs[0, idx].set_title(f'Uniform - {num_clusters} Clusters')
        axs[0, idx].set_xlabel('Distribution Index')
        axs[0, idx].set_ylabel('Semantic Entropy')
        axs[0, idx].set_xticks(range(len(se_values)))
        axs[0, idx].set_xticklabels([str(dist) for dist in uniform_distributions[num_clusters]], rotation=45)

        # Plot for skewed distribution
        se_values = [discrete_semantic_entropy(dist) for dist in skewed_distributions[num_clusters]]
        axs[1, idx].bar(range(len(se_values)), se_values, color='r')
        axs[1, idx].set_title(f'Skewed - {num_clusters} Clusters')
        axs[1, idx].set_xlabel('Distribution Index')
        axs[1, idx].set_ylabel('Semantic Entropy')
        axs[1, idx].set_xticks(range(len(se_values)))
        axs[1, idx].set_xticklabels([str(dist) for dist in skewed_distributions[num_clusters]], rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file)

if __name__ == '__main__':
    fire.Fire(generate_se_plots)