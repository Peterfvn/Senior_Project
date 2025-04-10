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

def test_ml(df):
    """
    Test the machine learning model on the data. This is baseline testing
    """
    lg = LogisticRegression(max_iter=1000)
    scores = cross_val_score(lg, df.iloc[:, 5:], df.iloc[:, 4], cv=5)
    print("Accuracy (Press prediction):", np.mean(scores))

def main():
    rat_id = 1.0
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

    for t in range(100):
        print(f"Time {t}: Neuron {neuron_map[best_neuron_at_time[t]]} with |t| {max_t_each_time[t]:.2f}")
    
    df_rat = df_rat[df_rat[df_rat.columns[3]] == 0.0]
    test_ml(df_rat)

    # top_neurons = np.argsort(-max_t_each_time)[:]
    # print(f"Top Neurons: {top_neurons}")
    # print(f"Neuron Map: {neuron_map}")
main()