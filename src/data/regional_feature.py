import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance

from load import *

def extract_regional_features(times):
    pre_tone = times[:, :20]
    tone = times[:, 20:80]
    post_tone = times[:, 80:]

    def compute_features(data):
        mean = np.mean(data, axis=1)
        std = np.std(data, axis=1)
        max_val = np.max(data, axis=1)
        min_val = np.min(data, axis=1)
        slope = (data[:, -1] - data[:, 0]) / data.shape[1]

        return np.stack([mean, std, max_val, min_val, slope], axis=1)
    
    pre_tone_features = compute_features(pre_tone)
    tone_features = compute_features(tone)
    post_tone_features = compute_features(post_tone)

    phase_features = np.concatenate([pre_tone_features, tone_features, post_tone_features], axis=1)
    return phase_features

def simple_classifier(features, labels):
    """Simple classifier for demonstration purposes"""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    clf = LogisticRegression(random_state=42)
    # clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return clf, X_test, y_test, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def analyze_feature_importance(classifier, X_test, y_test, feature_names):
    """Computes feature importance using coefficients and permutation importance."""
    # Coefficients from Logistic Regression
    coef_importance = classifier.coef_.flatten()
    
    # Permutation Importance
    perm_importance = permutation_importance(classifier, X_test, y_test, n_repeats=10, random_state=42)
    perm_values = perm_importance.importances_mean
    
    # Ensure the feature order is consistent across both plots
    sorted_idx = np.argsort(np.abs(coef_importance))[::-1]
    sorted_feature_names = np.array(feature_names)[sorted_idx]
    perm_values_sorted = perm_values[sorted_idx]
    coef_importance_sorted = coef_importance[sorted_idx]
    
    # Plot Feature Importance
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Permutation Importance Plot
    ax[0].barh(sorted_feature_names, perm_values_sorted, color='teal')
    ax[0].set_xlabel("Permutation Importance")
    ax[0].set_title("Permutation Feature Importance (Logistic Regression)")
    ax[0].invert_yaxis()
    
    # Model Coefficients Plot
    ax[1].barh(sorted_feature_names, coef_importance_sorted, color='darkorange')
    ax[1].set_xlabel("Model Coefficients")
    ax[1].set_title("Logistic Regression Coefficients")
    ax[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig("logistic_feature_importance.png")
    plt.close()


if __name__ == "__main__":
    # Load data
    df = load_file("VTA_con_4.csv")
    df = clean_data(df)
    df = trial_avg(df)

    df = df[df[df.columns[3]] == 0].reset_index(drop=True)

    labels = df.iloc[:, 4].values

    times = df.iloc[:, 5:].values
    phase_feats = extract_regional_features(times)

    feature_names = [f"phase_{i}_{stat}" for i in range(3) for stat in ['mean', 'var', 'max', 'min', 'slope']]

    classifier, X_test, y_test, results = simple_classifier(phase_feats, labels)
    analyze_feature_importance(classifier, X_test, y_test, feature_names)
    print(results)