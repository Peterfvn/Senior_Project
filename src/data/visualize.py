import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
from dotenv import load_dotenv
import os
from scipy.cluster.hierarchy import dendrogram, linkage

load_dotenv()
save_dir = os.getenv("VISUALIZATION_DIR")

def plot_population_avg(df, name):
    file = os.path.join(save_dir, name)

    num_rats = df[df.columns[0]].unique()
    fig, axes = plt.subplots(len(num_rats), 1, figsize=(10, 4 * len(num_rats)))
    fig.suptitle('Population Average Spike Count Over Time')

    for idx, rat in enumerate(num_rats):
        data = df[df[df.columns[0]] == rat]
        time = range(len(data.columns[5:]))

        for _, row in data.iterrows():
            axes[idx].plot(time, row[5:], label=f'Rat {rat} Group {row[1]}')

        axes[idx].set_xlabel('Time (ms)')
        axes[idx].set_ylabel('Activity')
        axes[idx].set_title(f'Rat {rat}')
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(file)
    plt.close()
    return

def plot_rat_clusters(df, clusters, name):
    fig, axes = plt.subplots(len(set(clusters.values())), 1, figsize=(10, 4 * len(set(clusters.values()))))

    for cluster_id in set(clusters.values()):
        cluster_rats =  [rat for rat, cluster in clusters.items() if cluster == cluster_id]
        cluster_data = df[df[df.columns[0]].isin(cluster_rats)]
        time = range(len(cluster_data.columns[5:]))

        axes[cluster_id].set_title(f'Cluster {cluster_id} Responses')
        for rat in cluster_rats:
            rat_data = cluster_data[cluster_data[cluster_data.columns[0]] == rat]
            axes[cluster_id].plot(time, rat_data.iloc[:, 5:].mean(), label=f'Rat {rat}')

        axes[cluster_id].set_xlabel('Time (ms)')
        axes[cluster_id].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, name))
    plt.close()

def visualize_clusters(df, clusters, time_cols, region_start_idx=None, region_end_idx=None, name=None):
    """Visualize time series data grouped by clusters."""
    unique_clusters = set(clusters.values())
    
    if region_start_idx is None or region_end_idx is None:
        # Use all time columns if no specific region is specified
        region_time_cols = time_cols
    else:
        # Use only the columns for the specified region
        region_time_cols = time_cols[region_start_idx:region_end_idx]
    
    # Create a figure with subplots for each cluster
    fig, axes = plt.subplots(len(unique_clusters), 1, figsize=(12, 4*len(unique_clusters)))
    if len(unique_clusters) == 1:
        axes = [axes]
    
    for cluster_id in unique_clusters:
        # Get the rats in this cluster
        cluster_rats = [rat_id for rat_id, cluster in clusters.items() if cluster == cluster_id]
        
        ax = axes[cluster_id]
        ax.set_title(f'Cluster {cluster_id} Responses')
        
        # Plot each rat's time series in this cluster
        for rat_id in cluster_rats:
            rat_data = df[df[df.columns[0]] == rat_id][region_time_cols].values.flatten()
            ax.plot(rat_data, label=f'Rat {rat_id}')
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Activity')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, name))
    plt.close()
    # return fig

def visualize_dendrogram(dtw_matrix, rat_ids, name):
    """Visualize hierarchical clustering as a dendrogram."""
    # Convert distance matrix to condensed form for linkage
    condensed_dist = []
    for i in range(len(dtw_matrix)):
        for j in range(i+1, len(dtw_matrix)):
            condensed_dist.append(dtw_matrix[i, j])
    
    # Compute linkage matrix
    Z = linkage(condensed_dist, method='average')
    
    # Plot dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=[f'Rat {rat}' for rat in rat_ids])
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Rat ID')
    plt.ylabel('DTW Distance')
    plt.savefig(os.path.join(save_dir, name))
    plt.close()