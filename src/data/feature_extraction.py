import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from load import *

def extract_general_features(df):
    """Extract general features from the DataFrame"""

    # Exclude non-time columns
    df = df.drop(columns=[df.columns[0], df.columns[1], df.columns[2], df.columns[3], df.columns[4]])
    
    features = []
    for _, row in df.iterrows():
        trial_features = {}
        # Statistics features
        trial_features['mean'] = np.mean(row)
        trial_features['std'] = np.std(row)
        trial_features['mad'] = stats.median_abs_deviation(row)

        # Peak based features
        peaks, properties = signal.find_peaks(row, height=np.mean(row) + np.std(row))
        trial_features['num_peaks'] = len(peaks)
        trial_features['mean_peak_height'] = np.mean(properties['peak_heights']) if len(peaks) > 0 else 0
        trial_features["latency_to_first_peak"] = peaks[0] if len(peaks) > 0 else np.nan

        # Temporal Dynamics
        trial_features['max_slope'] = np.max(np.diff(row))
        trial_features['pre_post_ratio'] = np.mean(row[20:80]) / np.mean(row[:20] + 1e-6)

        # Frequency Features
        fft_vals = np.abs(np.fft.rfft(row))
        fft_freqs = np.fft.rfftfreq(len(row), d=1/20)  # 20 Hz sampling rate
        trial_features["dominant_freq"] = fft_freqs[np.argmax(fft_vals)]
        trial_features["low_freq_power"] = np.sum(fft_vals[fft_freqs < 5])  # Power below 5 Hz

        features.append(trial_features)
    
    return pd.DataFrame(features)

def visualize_features(df_features):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i, feature in enumerate(df_features.columns[:-1]):
        axes[i].hist([df_features[feature][df_features["label"] == 0],
                      df_features[feature][df_features["label"] == 1]], bins=20, label=["No Press", "Press"], alpha=0.7)
        axes[i].set_title(feature)
    
    # Add global legend
    handles = [plt.Rectangle((0,0),1,1, color="blue", alpha=0.8), plt.Rectangle((0,0),1,1, color="orange", alpha=0.8)]
    labels = ["No Press", "Press"]
    fig.legend(handles, labels, loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.savefig("R1Norm_feature_distributions.png")
    plt.close()

def PCA_analysis(scaled_features, labels, name):
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=labels, cmap='coolwarm', alpha=1.0)
    plt.colorbar(label='Label')
    plt.title('PCA of Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(name)
    plt.close()

def tSNE_analysis(x_scaled, labels, name):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(x_scaled)
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='coolwarm', alpha=1.0)
    plt.colorbar(label='Label')
    plt.title('t-SNE of Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig(name)
    plt.close()

def UMAP_analysis(x_scaled, labels, name):
    import umap.umap_ as umap
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_result = umap_model.fit_transform(x_scaled)

    plt.figure(figsize=(10, 6))
    plt.scatter(umap_result[:, 0], umap_result[:, 1], c=labels, cmap='coolwarm', alpha=1.0)
    plt.colorbar(label='Label')
    plt.title('UMAP of Features')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.savefig(name)
    plt.close()

def normalize_by_rat_features(df):
    df_norm = df.copy()
    for rat in df["rat"].unique():
        mask = df["rat"] == rat
        feature_columns = [col for col in df.columns if col not in ["rat", "label"]]
        df_norm.loc[mask, feature_columns] = StandardScaler().fit_transform(df.loc[mask, feature_columns])
    return df_norm

def normalize_by_rat(df):
    df_norm = df.copy()
    for rat in df[df.columns[0]].unique():
        mask = df[df.columns[0]] == rat
        df_norm.loc[mask, df.columns[5:]] = StandardScaler().fit_transform(df.loc[mask, df.columns[5:]])
    return df_norm

def plotting(df):
    # Look at only R3 
    df = df[df[df.columns[0]] == 3]
    df = df.reset_index(drop=True)
    df_labels = df[df.columns[4]]

    # Extract features
    features = extract_general_features(df)
    features["label"] = df_labels

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.drop(columns=["label"]))
    X_scaled_important = scaler.fit_transform(features[['mad', 'std']])

    visualize_features(features)
    PCA_analysis(X_scaled, df_labels, "R3_PCA_features.png")

    tSNE_analysis(X_scaled, df_labels, "R3_tSNE_features.png")
    tSNE_analysis(X_scaled_important, df_labels, "R3_tSNE_important_features.png")

    UMAP_analysis(X_scaled, df_labels, "R3_UMAP_features.png")
    UMAP_analysis(X_scaled_important, df_labels, "R3_UMAP_important_features.png")

def main():
    """Train a simple logistic regression model using the feature extraction"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.model_selection import train_test_split

    df = load_file('PFC_con_4.csv')
    df = clean_data(df)
    df = trial_avg(df)

    df_labels = df[df.columns[4]]
    df_features = extract_general_features(df)
    df_features["label"] = df_labels

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features.drop(columns=["label"]))
    X_scaled_important = scaler.fit_transform(df_features[['mean', 'latency_to_first_peak']])

    X = X_scaled
    y = df_features["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()


    # df = load_file('PFC_con_4.csv')
    # df = clean_data(df)
    # df = trial_avg(df)

    # rats = df.iloc[:, 0]
    # df_labels = df[df.columns[4]]

    # norm_df = df[df[df.columns[0]] == 1]
    # # norm_df = normalize_by_rat(norm_df)
    # df_labels = norm_df[norm_df.columns[4]]

    # features = extract_general_features(norm_df)
    # features["label"] = df_labels
    # # features.insert(0, "rat", rats)

    # # No longer normalizing after feature extraction
    # # features = normalize_by_rat(features)

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(features.drop(columns=["label"]))
    # X_scaled_important = scaler.fit_transform(features[['mean', 'latency_to_first_peak']])

    # feature_columns = [col for col in features.columns if col not in ["rat", "label"]]
    # significant_features = []
    # for feature in feature_columns:
    #     t_stat, p_value = stats.ttest_ind(features[feature][features["label"] == 0],
    #                                        features[feature][features["label"] == 1], nan_policy='omit')
    #     if p_value < 0.05:
    #         significant_features.append((feature, p_value))

    # print("Significant Features (p < 0.05):")
    # for feat, p_val in significant_features:
    #     print(f"{feat}: p-value = {p_val:.4f}")