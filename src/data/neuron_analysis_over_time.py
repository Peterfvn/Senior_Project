from load import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

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
    
    # Divide the data by cue type
    cues = df[df.columns[3]].unique()
    time = np.arange(0, 100)

    for i, cue in enumerate(cues):
        # Filter the data for the current cue
        df_cue = df[df[df.columns[3]] == cue]

        # Plot the line plot
        mean = df_cue.iloc[:, 5:].mean(axis=0)
        std = df_cue.iloc[:, 5:].std(axis=0)

        plt.plot(time, mean, color=colors[i], label=f'{cue} trials')
        plt.fill_between(time, mean - std, mean + std, color=colors[i], alpha=0.2)

    # Add labels and title
    plt.xlabel("Time")
    plt.ylabel("Activity")
    plt.title(f"Neuron {neuron_id}")
    plt.legend()
    plt.savefig(f"lineplot_neuron_{neuron_id}.png")
    plt.close()  # Close the figure to free memory

    


def main():
    df = preparedata()
    rats = df[df.columns[0]].unique()

    # Just use rat 1 for now
    df_rat = df[df[df.columns[0]] == 1.0]

    # Count amount of neurons, create dict for channels
    channel_counts = defaultdict(int)
    num_neurons = len(df_rat) // 100
    
    for i in range(num_neurons):
        channel = df_rat.iloc[i*100, 1]
        count = channel_counts[channel]
        suffix = chr(ord('a') + count)  # Create a suffix like 'a', 'b', 'c', ...
        neuron_id = f"{channel}{suffix}"

        channel_counts[channel] += 1

        # Create lineplot for each neuron
        df_neuron = df_rat.iloc[i*100:(i+1)*100, :]
        line_plot_single_neuron(df_neuron, neuron_id=neuron_id)

main()