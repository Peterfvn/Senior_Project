import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
from dotenv import load_dotenv
import os

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