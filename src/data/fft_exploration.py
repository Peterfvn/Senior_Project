import numpy as num
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

from load import *

def fft_features(data):
    """
    Compute FFT features for the given data. Currently only doing normalized energy.
    """
    bin_size = 20
    windows = 100 // bin_size

    fft_energies = []

    for index, trial in data.iterrows():
        window_energy = []
        trial_data = trial.iloc[5:].values

        for i in range(windows):
            start = i * bin_size
            end = start + bin_size
            window = trial_data[start:end]

            fft_vals = fft(window)
            energy = np.sum(np.abs(fft_vals)**2)
            window_energy.append(energy)
        
        normalization = window_energy[0]
        rel_energy = [e - normalization for e in window_energy]
        fft_energies.append(rel_energy)

    return np.array(fft_energies)


def sliding_window_fft(df, window_size=20, stride=2):
    """Use a sliding window instead of disjoint windows"""
    all_features = []

    for index, trial in df.iterrows():
        trial_data = trial.iloc[5:].values
        num_bins = len(trial_data)

        features = []
        first_window_energy = None

        for start in range(0, num_bins - window_size + 1, stride):
            window = trial_data[start:start + window_size]

            fft_vals = fft(window)
            fft_half = fft_vals[:window_size // 2]
            energy = np.sum(np.abs(fft_half)**2)

            if first_window_energy is None:
                first_window_energy = energy
            normalized_energy = energy - first_window_energy

            features.append(normalized_energy)

        all_features.append(features)
    
    return np.array(all_features)

def plot_fft(data, labels, rat):
    """
    Plot FFT features for the given rat.
    """
    umap_embedder = umap.UMAP(n_components=2, random_state=42)
    umap_result = umap_embedder.fit_transform(data)

    tsne_result = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(data)

    pca_result = PCA(n_components=2).fit_transform(data)

    def plot_embedding(result, title):
        plt.figure(figsize=(10, 6))
        plt.scatter(result[:, 0], result[:, 1], c=labels, cmap='coolwarm', alpha=1.0)
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.savefig(title)
        plt.close()
    
    plot_embedding(umap_result, f"R{rat}_UMAP_SlidingFFT_Energy_Normalization.png")
    plot_embedding(tsne_result, f"R{rat}_TSNE_FFT_SlidingEnergy_Normalization.png")
    plot_embedding(pca_result, f"R{rat}_PCA_FFT_SlidingEnergy_Normalization.png")

def simple_classifier(data, labels):
    """
    Train a simple classifier on the FFT features.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    clf = LogisticRegression(max_iter=5000)
    scores = cross_val_score(clf, data, labels, cv=5)
    print("Accuracy (Press prediction):", np.mean(scores))

    
def main():
    data = load_file("VTA_con_4.csv")
    data = clean_data(data)
    data = trial_avg(data)

    rats = data[data.columns[0]].unique()

    fft_vals = []
    for rat in rats:
        rat_df = data[data[data.columns[0]] == rat]
        rat_df_negs = rat_df[rat_df[rat_df.columns[3]] == 0]
        # rat_df_negs = rat_df
        labels = rat_df_negs[rat_df_negs.columns[4]].reset_index(drop=True)

        features = sliding_window_fft(rat_df_negs)
        print(f"Rat: {rat}")
        simple_classifier(features, labels)

if __name__ == "__main__":
    main()