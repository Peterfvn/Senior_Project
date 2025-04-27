import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance

from load import *

def extract_features(data):
    """Extract phase-based features from the DataFrame"""
    regions = {
        'pre_tone': (0, 20),
        'tone': (20, 80),
        'post_tone': (80, 100)
    }

    pre_tone = data[:, :20]
    tone = data[:, 20:80]
    post_tone = data[:, 80:]

    def compute_features(data):
        # Basic stats
        mean = np.mean(data, axis=1)
        std = np.std(data, axis=1)
        max_val = np.max(data, axis=1)
        min_val = np.min(data, axis=1)
        slope = (data[:, -1] - data[:, 0]) / data.shape[1]

        # Temporal Features
        skewness = skew(data, axis=1)
        kurtosis_val = kurtosis(data, axis=1)

        # Frequency Features
        freqs, psd = welch(data, fs=20, nperseg=len(data[0]))
        mean_psd = np.mean(psd, axis=1)
        max_psd = np.max(psd, axis=1)

        # # Entropy Measures
        # def compute_shannon_entropy(data):
        #     """Compute Shannon entropy from a numerical sequence."""
        #     # Create a histogram with 10 bins
        #     hist, _ = np.histogram(data, bins=10, density=True)
        #     hist = hist[hist > 0]  # Remove zero values to avoid log(0) issues
        #     return entropy(hist)
        
        # entropy_val = compute_shannon_entropy(data)

        return np.stack([mean, std, max_val, min_val, slope, skewness, kurtosis_val, mean_psd, max_psd], axis=1)

    # Cross regional differences
    pre_tone_mean = np.mean(pre_tone)
    tone_mean = np.mean(tone)
    post_tone_mean = np.mean(post_tone)

    # cross_1_mean = tone_mean - pre_tone_mean
    # cross_2_mean = post_tone_mean - tone_mean

    pre_tone_features = compute_features(pre_tone)
    tone_features = compute_features(tone)
    post_tone_features = compute_features(post_tone)

    phase_features = np.concatenate([pre_tone_features, tone_features, post_tone_features], axis=1)
    return phase_features

def train_model(data, labels, model="lg", feature_elim=False):
    """
    Train simple classifiers on the extracted features for demonstration purposes.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import Lasso
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    feature_types = ['mean', 'std', 'max', 'min', 'slope', 'skewness', 'kurtosis', 'mean_psd', 'max_psd']
    phases = ['pre_tone', 'tone', 'post_tone']
    feature_names = []
    for phase in phases:
        for feature in feature_types:
            feature_names.append(f"{phase}_{feature}")

    # Replace NaN values with 0
    data = np.nan_to_num(data, nan=0.0)
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    if model == "lg":
        clf = LogisticRegression(random_state=42, max_iter=1000)
        if feature_elim == True:
            rfe = RFE(clf, n_features_to_select=10)
            rfe.fit(X_train, y_train)

            selected_features = np.where(rfe.support_)[0]
            selected_feature_names = [feature_names[i] for i in selected_features]
            print("Selected features:")
            for i, feature_name in enumerate(selected_feature_names):
                print(f"{i+1}. {feature_name}")
            return

    else:
        clf = RandomForestClassifier(random_state=42)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    if model == "rf":
        importances = clf.feature_importances_
        # feature_names = [f"Feature {i}" for i in range(len(importances))]

        # Convert to DataFrame
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        # Display top features
        print(importance_df.head(10))

    if model == "lg":
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_train, y_train)
        print("Lasso Feature Weights:", lasso.coef_)

    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    correlation_matrix(X_train_df, model)
    PCA_analysis(X_train_df)
    plot_permutation_importance(clf, X_test, y_test, feature_names)

    return clf, X_test, y_test, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def plot_permutation_importance(clf, X_test, y_test, feature_names):
    # Compute permutation importance
    perm_importance = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)

    sorted_idx = perm_importance.importances_mean.argsort()
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx])
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.tight_layout(pad=1.5)
    plt.title("Feature Importance via Permutation")
    plt.savefig("permutation_importance.png")
    plt.close()


def correlation_matrix(X_train, model):
    # Compute correlation matrix
    corr_matrix = pd.DataFrame(X_train).corr()

    name = "corr_mat_" + model + ".png"

    # Create a mask for insignificant correlations
    mask = np.abs(corr_matrix) < 0.8

    # Plot heatmap
    plt.figure(figsize=(20, 15))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.savefig(name)
    plt.close()

def PCA_analysis(X_train):
    # Apply PCA
    pca = PCA(n_components=None)  # Keep all components
    X_pca = pca.fit_transform(X_train)

    # Plot explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.savefig("pca_explained_variance.png")
    plt.close()

def main():
    df = load_file('VTA_con_4.csv')
    df = clean_data(df)
    df = trial_avg(df)

    df = df[df[df.columns[3]] == 0].reset_index(drop=True)

    time_values = df.iloc[:, 5:].values
    tone_labels = df.iloc[:, 3].values
    press_labels = df.iloc[:, 4].values

    features = extract_features(time_values)

    # Run without extraction because i'm returning early
    clf, X_test, y_test, metrics = train_model(features, press_labels, model="lg", feature_elim=False)
    print("\n")
    # train_model(features, tone_labels, model="rf")

    print(f"Press Classifier Metrics: {metrics}")
    # print("Tone Classifier Metrics: ", metrics2)
    

if __name__ == "__main__":
    main()