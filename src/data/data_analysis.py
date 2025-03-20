import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from load import *

def heatmap_time(df):
    # Unique rats
    num_rats = df[df.columns[0]].unique()

    fig, axes = plt.subplots(len(num_rats), 2, figsize=(12, 6 * len(num_rats)))

    for i, rat in enumerate(num_rats):
        # Filter data for the current rat
        rat_data = df[df[df.columns[0]] == rat]

        # Split data by tone type
        vmin = np.percentile(rat_data.iloc[:, 5:], 5)
        vmax = np.percentile(rat_data.iloc[:, 5:], 95)

        positive_tone = rat_data[rat_data[rat_data.columns[3]] == 1]
        negative_tone = rat_data[rat_data[rat_data.columns[3]] == 0]

        sns.heatmap(positive_tone.iloc[:, 5:], cmap="coolwarm", ax=axes[i, 0], center=0, vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f'Positive Tone for Rat {rat}')

        sns.heatmap(negative_tone.iloc[:, 5:], cmap="coolwarm", ax=axes[i, 1], center=0, vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f'Negative Tone for Rat {rat}')

    plt.tight_layout()
    plt.savefig(f'heatmap_of_time.png')
    plt.close()

def pca_tsne(df):
    num_rats = df[df.columns[0]].unique()

    fig, axes = plt.subplots(len(num_rats)*2, 2, figsize=(12, 6 * len(num_rats)))

    for i, rat in enumerate(num_rats):
        rat_data = df[df[df.columns[0]] == rat]

        # PCA
        X = rat_data.iloc[:, 5:].values
        y_tone = rat_data[rat_data.columns[3]].values

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)

        # tSNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(X)

        



def main():
    df = load_file('PFC_con_4.csv')
    df = clean_data(df)
    df = trial_avg(df)

    heatmap_time(df)

if __name__ == "__main__":
    main()