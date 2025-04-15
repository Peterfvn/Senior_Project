from load import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

def preparedata():
    """
    Load and prepare the data. We are analyzing each neuron's activity over time for different trials.
    """
    df = load_file('PFC_con_4.csv') # This is all I'm doing I suppose
    return df

def heatmap_single_neuron(df):
    """
    Plot a heatmap of a single neuron's activity over time.
    """

    # Create the heatmap
    vmax = np.percentile(df.iloc[:, 5:], 99)
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(df.iloc[:, 5:], cmap='viridis', cbar=True, yticklabels=False, xticklabels=False, vmin=-30, vmax=vmax)

    # Add a dividing line between positive and negative trials
    ax.axhline(50, color='black', linewidth=2, linestyle='--')

    # Set labels and title
    plt.xlabel("Time")
    plt.ylabel("Trials")
    plt.title(f"Neuron {df.columns[1]}")
    plt.savefig(f"heatmap_neuron_{df.columns[1]}.png")
    plt.close()

def line_plot_single_neuron(df, neuron_id):
    """
    Plot a line plot of a single neuron's activity over time.
    """
    # Define two colors for the two trial types
    colors = ['blue', 'red']
    
    # Divide the data by press type
    press = df[df.columns[3]].unique()
    time = np.arange(0, 100)

    for i, press in enumerate(press):
        # Filter the data for the current press
        df_press = df[df[df.columns[3]] == press]

        # Plot the line plot
        mean = df_press.iloc[:, 5:].mean(axis=0)
        std = df_press.iloc[:, 5:].std(axis=0)

        if press == 1.0:
            press_label = 'Positive Tone'
        else:
            press_label = 'Negative Tone'

        plt.plot(time, mean, color=colors[i], label=f'{press_label} trials')
        plt.fill_between(time, mean - std, mean + std, color=colors[i], alpha=0.2)

    # Add labels and title
    plt.xlabel("Time")
    plt.ylabel("Activity")
    plt.title(f"Neuron {neuron_id}")
    plt.legend()
    plt.savefig(f"lineplot_neuron_{neuron_id}.png")
    plt.close()

def t_test_list(df, press_test):
    """
    Perform a t-test on the activity of the neuron across trials. Do this for each neuron to construct
    a matrix of results to determine which neurons are significant at what times.
    """

    t_stats = []
    if press_test:
        split_col = df.columns[4]
    else:
        split_col = df.columns[3]

    for i in range(100):
        # Split by cue type
        positive_trials = df[df[split_col] == 1.0]
        negative_trials = df[df[split_col] == 0.0]

        # Perform t-test
        if len(positive_trials) < 2 or len(negative_trials) < 2:
            t = 0.0
        else:
            try:
                t, p = ttest_ind(positive_trials.iloc[:, 5+i], negative_trials.iloc[:, 5+i], equal_var=False)
            except:
                print(f"Error on bin {i}")
                t = 0.0
        
        t_stats.append(t)
    
    return np.array(t_stats)

    
def t_test_heatmap(df):
    sns.heatmap(np.abs(df), cmap='coolwarm', cbar=True)
    plt.xlabel("Time Bins")
    plt.ylabel("Neurons")
    plt.title("Discriminability Heatmap (|t-stat|)")
    plt.savefig("t_test_heatmap.png")
    plt.close()

def build_best_neuron_per_time_feature_matrix(df_rat, t_matrix, num_neurons=22):
    best_neuron_at_time = np.argmax(np.abs(t_matrix), axis=0)  # shape (100,)
    
    feature_matrix = []
    labels = []
    n_trials = 50

    for trial_idx in range(n_trials):
        feature_row = []
        label = df_rat.iloc[n_trials + trial_idx, 4]
        labels.append(label)

        for t in range(100):
            neuron_idx = best_neuron_at_time[t]

            # Compute global column index for this neuron's bin t
            col_idx = 5 + t
            row_idx = neuron_idx * n_trials + trial_idx

            # Extract value for this trial
            value = df_rat.iloc[row_idx, col_idx]
            feature_row.append(value)

        feature_matrix.append(feature_row)

    feature_matrix = np.array(feature_matrix)  # shape: (50 trials, 100 time-based features)

    return feature_matrix, labels


def test_ml(df):
    """
    Test the machine learning model on the data. This is baseline testing
    """
    lg = LogisticRegression(max_iter=1000)
    scores = cross_val_score(lg, df.iloc[:, 5:], df.iloc[:, 4], cv=5)
    print("Accuracy (Press prediction):", np.mean(scores))

def discriminability_lg(data, labels):
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, data, labels, cv=5)
    print("Discriminability (Press prediction):", np.mean(scores))

def main():
    rat_id = 10.0
    plot = False
    press_test = True

    df = preparedata()
    rats = df[df.columns[0]].unique()
    
    # Just use rat 1 for now
    df_rat = df[df[df.columns[0]] == rat_id]

    # Count amount of neurons, create dict for channels
    channel_counts = defaultdict(int)
    neuron_map = defaultdict(list)
    
    num_neurons = len(df_rat) // 100
    
    # Create matrix for t-test
    t_matrix = np.zeros((num_neurons, 100))

    for i in range(num_neurons):
        channel = df_rat.iloc[i*100, 1]
        count = channel_counts[channel]
        suffix = chr(ord('a') + count)  # Create a suffix like 'a', 'b', 'c', ...
        neuron_id = f"{channel}{suffix}"

        channel_counts[channel] += 1
        neuron_map[i] = neuron_id

        # Create lineplot for each neuron
        df_neuron = df_rat.iloc[i*100:(i+1)*100, :]

        # Filter out all positive cue trials
        if plot:
            line_plot_single_neuron(df_neuron, neuron_id)
        
        if press_test:
            df_neuron = df_neuron[df_neuron[df_neuron.columns[3]] == 0.0]
        t_matrix[i, :] = t_test_list(df_neuron, press_test)
    
    max_t_each_time = np.max(np.abs(t_matrix), axis=0)
    best_neuron_at_time = np.argmax(np.abs(t_matrix), axis=0)

    df_rat = df_rat[df_rat[df_rat.columns[3]] == 0.0].reset_index(drop=True) 
    
    feature_vector, labels = build_best_neuron_per_time_feature_matrix(df_rat, t_matrix, num_neurons=22)
    discriminability_lg(feature_vector, labels)

def cross_validated_tstat_model(df_rat, num_neurons, k=5):
    n_trials = 50
    X_all = df_rat.iloc[:, 5:].values
    y_all = df_rat.iloc[:n_trials, 4].values  # labels repeat every 50 trials

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_trials))):
        # print(f"Fold {fold + 1}")

        # Compute t-matrix using only training trials
        t_matrix = np.zeros((num_neurons, 100))
        for neuron in range(num_neurons):
            df_neuron = df_rat.iloc[neuron * n_trials : (neuron + 1) * n_trials]
            df_train = df_neuron.iloc[train_idx]
            t_matrix[neuron] = t_test_list(df_train, press_test=True)

        # Select best neuron per time bin
        best_neuron_at_time = np.argmax(np.abs(t_matrix), axis=0)

        # Build feature vectors
        def extract_features(indices):
            features = []
            for trial_idx in indices:
                row = []
                for t in range(100):
                    neuron = best_neuron_at_time[t]
                    col_idx = 5  + t
                    row.append(df_rat.iloc[neuron * n_trials + trial_idx, col_idx])
                features.append(row)
            return np.array(features)

        X_train = extract_features(train_idx)
        y_train = y_all[train_idx]

        X_test = extract_features(test_idx)
        y_test = y_all[test_idx]

        # Train classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted')

        scores.append((acc, prec, rec, f1))

        # print(f"  Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

    scores = np.array(scores)
    print("==== Overall Performance ====")
    print(f"Accuracy: {scores[:,0].mean():.3f} ± {scores[:,0].std():.3f}")
    print(f"Precision: {scores[:,1].mean():.3f} ± {scores[:,1].std():.3f}")
    print(f"Recall: {scores[:,2].mean():.3f} ± {scores[:,2].std():.3f}")
    print(f"F1 Score: {scores[:,3].mean():.3f} ± {scores[:,3].std():.3f}\n")

def run_all_ml():
    df = load_file('PFC_con_4.csv')
    rats = df[df.columns[0]].unique()

    for rat_id in rats:
        rat_df = df[df[df.columns[0]] == rat_id].reset_index(drop=True)

        # Filter down to only negative tones
        rat_df = rat_df[rat_df[rat_df.columns[3]] == 0.0].reset_index(drop=True)

        num_neurons = len(rat_df) // 100

        print(f"Rat {rat_id} - Discriminability:")

        cross_validated_tstat_model(rat_df, num_neurons, k=5)

run_all_ml()

"""
Rat 1.0 - Discriminability:
Discriminability (Press prediction): 1.0
Rat 3.0 - Discriminability:
C:\\Users\\pfvan\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\sklearn\\model_selection\\_split.py:805: UserWarning: The least populated class in y has only 2 members, which is less than n_splits=5.
warnings.warn(
Discriminability (Press prediction): 1.0
Rat 5.0 - Discriminability:
Discriminability (Press prediction): 0.96
Rat 10.0 - Discriminability:
Discriminability (Press prediction): 0.9400000000000001
Rat 11.0 - Discriminability:
C:\\Users\\pfvan\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\sklearn\\model_selection\\_split.py:805: UserWarning: The least populated class in y has only 2 members, which is less than n_splits=5.
warnings.warn(
Discriminability (Press prediction): 0.96
Rat 15.0 - Discriminability:
Discriminability (Press prediction): 0.9400000000000001
Rat 16.0 - Discriminability:
Discriminability (Press prediction): 0.9800000000000001
"""

"""
Rat 1.0 - Discriminability:
==== Overall Performance ====
Accuracy: 0.880 ± 0.117
Precision: 0.788 ± 0.201
Recall: 0.880 ± 0.117
F1 Score: 0.828 ± 0.165

Rat 3.0 - Discriminability:
==== Overall Performance ====
Accuracy: 0.960 ± 0.049
Precision: 0.924 ± 0.093
Recall: 0.960 ± 0.049
F1 Score: 0.941 ± 0.072

Rat 5.0 - Discriminability:
==== Overall Performance ====
Accuracy: 0.680 ± 0.172
Precision: 0.690 ± 0.285
Recall: 0.680 ± 0.172
F1 Score: 0.641 ± 0.219

Rat 10.0 - Discriminability:
==== Overall Performance ====
Accuracy: 0.520 ± 0.075
Precision: 0.636 ± 0.100
Recall: 0.520 ± 0.075
F1 Score: 0.505 ± 0.105

Rat 11.0 - Discriminability:
==== Overall Performance ====
Accuracy: 0.960 ± 0.049
Precision: 0.924 ± 0.093
Recall: 0.960 ± 0.049
F1 Score: 0.941 ± 0.072

Rat 15.0 - Discriminability:
==== Overall Performance ====
Accuracy: 0.460 ± 0.102
Precision: 0.440 ± 0.171
Recall: 0.460 ± 0.102
F1 Score: 0.433 ± 0.115

Rat 16.0 - Discriminability:
==== Overall Performance ====
Accuracy: 0.780 ± 0.117
Precision: 0.708 ± 0.179
Recall: 0.780 ± 0.117
F1 Score: 0.738 ± 0.143
"""